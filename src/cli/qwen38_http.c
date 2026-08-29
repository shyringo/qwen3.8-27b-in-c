#include "qwen38_http.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define JSMN_STATIC
#include "jsmn.h"

static void q38_http_error(char *error, size_t capacity, const char *message)
{
    if (!error || capacity == 0) return;
    snprintf(error, capacity, "%s", message);
}

void q38_http_chat_request_init(Q38HttpChatRequest *request)
{
    if (request) memset(request, 0, sizeof(*request));
}

void q38_http_chat_request_free(Q38HttpChatRequest *request)
{
    if (!request) return;
    free(request->model);
    for (size_t i = 0; i < request->message_count; ++i) {
        free(request->messages[i].role);
        free(request->messages[i].content);
        free(request->messages[i].tool_call_id);
        for (size_t j = 0; j < request->messages[i].tool_call_count; ++j) {
            free(request->messages[i].tool_calls[j].id);
            free(request->messages[i].tool_calls[j].name);
            free(request->messages[i].tool_calls[j].arguments);
        }
    }
    for (size_t i = 0; i < request->tool_count; ++i) {
        free(request->tools[i].name);
        free(request->tools[i].function_json);
    }
    memset(request, 0, sizeof(*request));
}

void q38_buffer_init(Q38Buffer *buffer)
{
    if (buffer) memset(buffer, 0, sizeof(*buffer));
}

void q38_buffer_free(Q38Buffer *buffer)
{
    if (!buffer) return;
    free(buffer->data);
    memset(buffer, 0, sizeof(*buffer));
}

int q38_buffer_append(Q38Buffer *buffer, const char *data, size_t length)
{
    if (!buffer || (!data && length)) return 0;
    if (length > SIZE_MAX - buffer->length - 1u) return 0;
    const size_t needed = buffer->length + length + 1u;
    if (needed > buffer->capacity) {
        size_t capacity = buffer->capacity ? buffer->capacity : 256u;
        while (capacity < needed) {
            if (capacity > SIZE_MAX / 2u) {
                capacity = needed;
                break;
            }
            capacity *= 2u;
        }
        char *larger = (char *)realloc(buffer->data, capacity);
        if (!larger) return 0;
        buffer->data = larger;
        buffer->capacity = capacity;
    }
    if (!buffer->data) return 0;
    if (length) memcpy(buffer->data + buffer->length, data, length);
    buffer->length += length;
    buffer->data[buffer->length] = '\0';
    return 1;
}

int q38_buffer_append_string(Q38Buffer *buffer, const char *text)
{
    return text && q38_buffer_append(buffer, text, strlen(text));
}

int q38_buffer_append_json_string(Q38Buffer *buffer,
                                  const char *text, size_t length)
{
    if (!q38_buffer_append(buffer, "\"", 1u)) return 0;
    for (size_t i = 0; i < length; ++i) {
        const unsigned char c = (unsigned char)text[i];
        const char *escape = NULL;
        switch (c) {
        case '"': escape = "\\\""; break;
        case '\\': escape = "\\\\"; break;
        case '\b': escape = "\\b"; break;
        case '\f': escape = "\\f"; break;
        case '\n': escape = "\\n"; break;
        case '\r': escape = "\\r"; break;
        case '\t': escape = "\\t"; break;
        default: break;
        }
        if (escape) {
            if (!q38_buffer_append_string(buffer, escape)) return 0;
        } else if (c < 0x20u) {
            char encoded[7];
            snprintf(encoded, sizeof(encoded), "\\u%04x", c);
            if (!q38_buffer_append(buffer, encoded, 6u)) return 0;
        } else if (!q38_buffer_append(buffer, (const char *)&text[i], 1u)) {
            return 0;
        }
    }
    return q38_buffer_append(buffer, "\"", 1u);
}

static int q38_utf8_sequence(const unsigned char *text, size_t available,
                             size_t *length)
{
    const unsigned char first = text[0];
    if (first <= 0x7fu) {
        *length = 1u;
        return 1;
    }
    size_t needed = 0;
    if (first >= 0xc2u && first <= 0xdfu) needed = 2u;
    else if (first >= 0xe0u && first <= 0xefu) needed = 3u;
    else if (first >= 0xf0u && first <= 0xf4u) needed = 4u;
    else return -1;
    if (available < 2u) return 0;
    const unsigned char second = text[1];
    if ((second & 0xc0u) != 0x80u ||
        (first == 0xe0u && second < 0xa0u) ||
        (first == 0xedu && second >= 0xa0u) ||
        (first == 0xf0u && second < 0x90u) ||
        (first == 0xf4u && second >= 0x90u))
        return -1;
    if (available < needed) return 0;
    for (size_t i = 2u; i < needed; ++i)
        if ((text[i] & 0xc0u) != 0x80u) return -1;
    *length = needed;
    return 1;
}

void q38_utf8_stream_init(Q38Utf8Stream *stream,
                          int (*emit)(void *context,
                                      const char *data, size_t length),
                          void *context)
{
    if (!stream) return;
    q38_buffer_init(&stream->pending);
    stream->emit = emit;
    stream->context = context;
}

void q38_utf8_stream_free(Q38Utf8Stream *stream)
{
    if (!stream) return;
    q38_buffer_free(&stream->pending);
    stream->emit = NULL;
    stream->context = NULL;
}

int q38_utf8_stream_write(void *context, const char *data, size_t length)
{
    Q38Utf8Stream *stream = (Q38Utf8Stream *)context;
    if (!stream || !stream->emit || !q38_buffer_append(&stream->pending,
                                                        data, length))
        return 0;
    static const char replacement[] = "\xef\xbf\xbd";
    size_t cursor = 0;
    size_t run = 0;
    while (cursor < stream->pending.length) {
        size_t sequence = 0;
        const int status = q38_utf8_sequence(
            (const unsigned char *)stream->pending.data + cursor,
            stream->pending.length - cursor, &sequence);
        if (status == 0) break;
        if (status > 0) {
            cursor += sequence;
            continue;
        }
        if (cursor > run && !stream->emit(stream->context,
                                          stream->pending.data + run,
                                          cursor - run))
            return 0;
        if (!stream->emit(stream->context, replacement,
                          sizeof(replacement) - 1u))
            return 0;
        ++cursor;
        run = cursor;
    }
    if (cursor > run && !stream->emit(stream->context,
                                      stream->pending.data + run,
                                      cursor - run))
        return 0;
    const size_t remaining = stream->pending.length - cursor;
    if (remaining)
        memmove(stream->pending.data, stream->pending.data + cursor, remaining);
    stream->pending.length = remaining;
    if (stream->pending.data) stream->pending.data[remaining] = '\0';
    return 1;
}

int q38_utf8_stream_flush(Q38Utf8Stream *stream)
{
    static const char replacement[] = "\xef\xbf\xbd";
    if (!stream || !stream->emit) return 0;
    if (stream->pending.length &&
        !stream->emit(stream->context, replacement, sizeof(replacement) - 1u))
        return 0;
    stream->pending.length = 0;
    if (stream->pending.data) stream->pending.data[0] = '\0';
    return 1;
}

static int q38_token_equal(const char *json, const jsmntok_t *token,
                           const char *text)
{
    const size_t length = strlen(text);
    return token->type == JSMN_STRING &&
           token->end >= token->start &&
           (size_t)(token->end - token->start) == length &&
           memcmp(json + token->start, text, length) == 0;
}

static int q38_token_next(const jsmntok_t *tokens, int count, int index)
{
    if (index < 0 || index >= count) return count;
    const int end = tokens[index].end;
    ++index;
    while (index < count && tokens[index].start < end) ++index;
    return index;
}

static int q38_object_get(const char *json, const jsmntok_t *tokens,
                          int count, int object, const char *key)
{
    if (object < 0 || object >= count || tokens[object].type != JSMN_OBJECT)
        return -1;
    int index = object + 1;
    while (index < count && tokens[index].start < tokens[object].end) {
        const int value = index + 1;
        if (value >= count || tokens[value].start >= tokens[object].end)
            return -1;
        if (q38_token_equal(json, &tokens[index], key)) return value;
        index = q38_token_next(tokens, count, value);
    }
    return -1;
}

static int q38_hex(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int q38_utf8(Q38Buffer *output, uint32_t codepoint)
{
    char bytes[4];
    size_t count = 0;
    if (codepoint <= 0x7fu) {
        bytes[count++] = (char)codepoint;
    } else if (codepoint <= 0x7ffu) {
        bytes[count++] = (char)(0xc0u | (codepoint >> 6));
        bytes[count++] = (char)(0x80u | (codepoint & 0x3fu));
    } else if (codepoint <= 0xffffu) {
        if (codepoint >= 0xd800u && codepoint <= 0xdfffu) return 0;
        bytes[count++] = (char)(0xe0u | (codepoint >> 12));
        bytes[count++] = (char)(0x80u | ((codepoint >> 6) & 0x3fu));
        bytes[count++] = (char)(0x80u | (codepoint & 0x3fu));
    } else if (codepoint <= 0x10ffffu) {
        bytes[count++] = (char)(0xf0u | (codepoint >> 18));
        bytes[count++] = (char)(0x80u | ((codepoint >> 12) & 0x3fu));
        bytes[count++] = (char)(0x80u | ((codepoint >> 6) & 0x3fu));
        bytes[count++] = (char)(0x80u | (codepoint & 0x3fu));
    } else {
        return 0;
    }
    return q38_buffer_append(output, bytes, count);
}

static char *q38_decode_string(const char *json, const jsmntok_t *token)
{
    if (!json || !token || token->type != JSMN_STRING ||
        token->start < 0 || token->end < token->start)
        return NULL;
    Q38Buffer output;
    q38_buffer_init(&output);
    for (int i = token->start; i < token->end; ++i) {
        const unsigned char c = (unsigned char)json[i];
        if (c != '\\') {
            if (c < 0x20u || !q38_buffer_append(&output, (const char *)&json[i], 1u))
                goto fail;
            continue;
        }
        if (++i >= token->end) goto fail;
        const char escaped = json[i];
        char decoded = 0;
        switch (escaped) {
        case '"': decoded = '"'; break;
        case '\\': decoded = '\\'; break;
        case '/': decoded = '/'; break;
        case 'b': decoded = '\b'; break;
        case 'f': decoded = '\f'; break;
        case 'n': decoded = '\n'; break;
        case 'r': decoded = '\r'; break;
        case 't': decoded = '\t'; break;
        case 'u': {
            if (i + 4 >= token->end) goto fail;
            uint32_t codepoint = 0;
            for (int j = 0; j < 4; ++j) {
                const int digit = q38_hex(json[++i]);
                if (digit < 0) goto fail;
                codepoint = (codepoint << 4) | (uint32_t)digit;
            }
            if (codepoint >= 0xd800u && codepoint <= 0xdbffu) {
                if (i + 6 >= token->end || json[i + 1] != '\\' ||
                    json[i + 2] != 'u') goto fail;
                i += 2;
                uint32_t low = 0;
                for (int j = 0; j < 4; ++j) {
                    const int digit = q38_hex(json[++i]);
                    if (digit < 0) goto fail;
                    low = (low << 4) | (uint32_t)digit;
                }
                if (low < 0xdc00u || low > 0xdfffu) goto fail;
                codepoint = 0x10000u + ((codepoint - 0xd800u) << 10) +
                            (low - 0xdc00u);
            }
            if (!q38_utf8(&output, codepoint)) goto fail;
            continue;
        }
        default: goto fail;
        }
        if (!q38_buffer_append(&output, &decoded, 1u)) goto fail;
    }
    if (!output.data) {
        output.data = (char *)calloc(1u, 1u);
        if (!output.data) return NULL;
    }
    return output.data;
fail:
    q38_buffer_free(&output);
    return NULL;
}

static int q38_parse_u64(const char *json, const jsmntok_t *token,
                         uint64_t *value)
{
    if (!token || token->type != JSMN_PRIMITIVE || token->end <= token->start)
        return 0;
    const size_t length = (size_t)(token->end - token->start);
    if (length >= 64u) return 0;
    char text[64];
    memcpy(text, json + token->start, length);
    text[length] = '\0';
    char *end = NULL;
    errno = 0;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || end == text || *end || text[0] == '-') return 0;
    *value = (uint64_t)parsed;
    return 1;
}

static int q38_parse_float(const char *json, const jsmntok_t *token,
                           float *value)
{
    if (!token || token->type != JSMN_PRIMITIVE || token->end <= token->start)
        return 0;
    const size_t length = (size_t)(token->end - token->start);
    if (length >= 64u) return 0;
    char text[64];
    memcpy(text, json + token->start, length);
    text[length] = '\0';
    char *end = NULL;
    errno = 0;
    const float parsed = strtof(text, &end);
    if (errno || end == text || *end || !isfinite(parsed)) return 0;
    *value = parsed;
    return 1;
}

static int q38_primitive_is(const char *json, const jsmntok_t *token,
                            const char *value)
{
    if (!token || token->type != JSMN_PRIMITIVE) return 0;
    const size_t length = strlen(value);
    return token->end >= token->start &&
           (size_t)(token->end - token->start) == length &&
           memcmp(json + token->start, value, length) == 0;
}

static int q38_supported_role(const char *role)
{
    return strcmp(role, "system") == 0 || strcmp(role, "developer") == 0 ||
           strcmp(role, "user") == 0 || strcmp(role, "assistant") == 0 ||
           strcmp(role, "tool") == 0;
}

static char *q38_copy_token(const char *json, const jsmntok_t *token)
{
    if (!token || token->start < 0 || token->end < token->start) return NULL;
    const size_t length = (size_t)(token->end - token->start);
    char *copy = (char *)malloc(length + 1u);
    if (!copy) return NULL;
    memcpy(copy, json + token->start, length);
    copy[length] = '\0';
    return copy;
}

static int q38_tool_name_valid(const char *name)
{
    const size_t length = name ? strlen(name) : 0u;
    if (!length || length > 128u) return 0;
    for (size_t i = 0; i < length; ++i) {
        const unsigned char c = (unsigned char)name[i];
        if (!(isalnum(c) || c == '_' || c == '-' || c == '.' || c == ':'))
            return 0;
    }
    return 1;
}

int q38_http_parse_chat_request(const char *json, size_t length,
                                Q38HttpChatRequest *request,
                                char *error, size_t error_capacity)
{
    if (!json || !request || length == 0 || length > Q38_HTTP_MAX_BODY) {
        q38_http_error(error, error_capacity, "request body is empty or too large");
        return 0;
    }
    q38_http_chat_request_init(request);
    unsigned int capacity = 256u;
    jsmntok_t *tokens = NULL;
    int count = JSMN_ERROR_NOMEM;
    while (count == JSMN_ERROR_NOMEM && capacity <= 16384u) {
        jsmntok_t *larger = (jsmntok_t *)realloc(tokens,
                                                  capacity * sizeof(*tokens));
        if (!larger) break;
        tokens = larger;
        jsmn_parser parser;
        jsmn_init(&parser);
        count = jsmn_parse(&parser, json, length, tokens, capacity);
        capacity *= 2u;
    }
    if (count < 1 || tokens[0].type != JSMN_OBJECT) {
        q38_http_error(error, error_capacity, "body must be one valid JSON object");
        free(tokens);
        return 0;
    }
    for (size_t i = (size_t)tokens[0].end; i < length; ++i) {
        if (json[i] != ' ' && json[i] != '\t' &&
            json[i] != '\r' && json[i] != '\n') {
            q38_http_error(error, error_capacity,
                           "body contains trailing JSON data");
            free(tokens);
            return 0;
        }
    }

    int token = q38_object_get(json, tokens, count, 0, "model");
    if (token < 0 || !(request->model = q38_decode_string(json, &tokens[token]))) {
        q38_http_error(error, error_capacity, "model must be a JSON string");
        goto fail;
    }
    if (strlen(request->model) > 128u) {
        q38_http_error(error, error_capacity, "model is too long");
        goto fail;
    }

    token = q38_object_get(json, tokens, count, 0, "stream");
    if (token >= 0) {
        if (q38_primitive_is(json, &tokens[token], "true")) request->stream = 1;
        else if (!q38_primitive_is(json, &tokens[token], "false")) {
            q38_http_error(error, error_capacity, "stream must be true or false");
            goto fail;
        }
    }
    token = q38_object_get(json, tokens, count, 0, "stream_options");
    if (token >= 0) {
        if (!request->stream || tokens[token].type != JSMN_OBJECT) {
            q38_http_error(error, error_capacity,
                           "stream_options requires stream true");
            goto fail;
        }
        const int include = q38_object_get(json, tokens, count, token,
                                           "include_usage");
        if (include >= 0) {
            if (q38_primitive_is(json, &tokens[include], "true"))
                request->include_usage = 1;
            else if (!q38_primitive_is(json, &tokens[include], "false")) {
                q38_http_error(error, error_capacity,
                               "include_usage must be true or false");
                goto fail;
            }
        }
    }
    token = q38_object_get(json, tokens, count, 0, "n");
    if (token >= 0) {
        uint64_t n = 0;
        if (!q38_parse_u64(json, &tokens[token], &n) || n != 1u) {
            q38_http_error(error, error_capacity, "n must be 1");
            goto fail;
        }
    }
    if (q38_object_get(json, tokens, count, 0, "response_format") >= 0) {
        q38_http_error(error, error_capacity,
                       "structured output is not supported");
        goto fail;
    }
    token = q38_object_get(json, tokens, count, 0, "tool_choice");
    if (token >= 0) {
        if (q38_token_equal(json, &tokens[token], "none"))
            request->tool_choice_none = 1;
        else if (!q38_token_equal(json, &tokens[token], "auto")) {
            q38_http_error(error, error_capacity,
                           "tool_choice currently supports only auto or none");
            goto fail;
        }
    }
    const int tools_token = q38_object_get(json, tokens, count, 0, "tools");
    if (tools_token >= 0) {
        if (tokens[tools_token].type != JSMN_ARRAY) {
            q38_http_error(error, error_capacity, "tools must be a JSON array");
            goto fail;
        }
        int tool_index = tools_token + 1;
        while (tool_index < count &&
               tokens[tool_index].start < tokens[tools_token].end) {
            if (tokens[tool_index].type != JSMN_OBJECT ||
                request->tool_count >= Q38_HTTP_MAX_TOOLS) {
                q38_http_error(error, error_capacity,
                               "tools contains too many or invalid items");
                goto fail;
            }
            const int type = q38_object_get(json, tokens, count,
                                            tool_index, "type");
            const int function = q38_object_get(json, tokens, count,
                                                tool_index, "function");
            if (type < 0 || function < 0 ||
                !q38_token_equal(json, &tokens[type], "function") ||
                tokens[function].type != JSMN_OBJECT) {
                q38_http_error(error, error_capacity,
                               "only function tools are supported");
                goto fail;
            }
            const int name = q38_object_get(json, tokens, count,
                                            function, "name");
            const int parameters = q38_object_get(json, tokens, count,
                                                  function, "parameters");
            Q38HttpTool *tool = &request->tools[request->tool_count];
            tool->name = name >= 0 ? q38_decode_string(json, &tokens[name]) : NULL;
            tool->function_json = q38_copy_token(json, &tokens[function]);
            ++request->tool_count;
            if (!q38_tool_name_valid(tool->name) || !tool->function_json ||
                parameters < 0 || tokens[parameters].type != JSMN_OBJECT) {
                q38_http_error(error, error_capacity,
                               "function tool needs a valid name and parameters schema");
                goto fail;
            }
            for (size_t i = 0; i + 1u < request->tool_count; ++i) {
                if (strcmp(request->tools[i].name, tool->name) == 0) {
                    q38_http_error(error, error_capacity,
                                   "function tool names must be unique");
                    goto fail;
                }
            }
            tool_index = q38_token_next(tokens, count, tool_index);
        }
    }

    token = q38_object_get(json, tokens, count, 0, "max_completion_tokens");
    if (token < 0) token = q38_object_get(json, tokens, count, 0, "max_tokens");
    if (token >= 0) {
        uint64_t value = 0;
        if (!q38_parse_u64(json, &tokens[token], &value) ||
            value == 0 || value > 65536u) {
            q38_http_error(error, error_capacity, "max_tokens is out of range");
            goto fail;
        }
        request->max_tokens = (uint32_t)value;
        request->has_max_tokens = 1;
    }
    token = q38_object_get(json, tokens, count, 0, "seed");
    if (token >= 0) {
        if (!q38_parse_u64(json, &tokens[token], &request->seed)) {
            q38_http_error(error, error_capacity, "seed must be a non-negative integer");
            goto fail;
        }
        request->has_seed = 1;
    }
    token = q38_object_get(json, tokens, count, 0, "temperature");
    if (token >= 0) {
        if (!q38_parse_float(json, &tokens[token], &request->temperature) ||
            request->temperature < 0.0f || request->temperature > 2.0f) {
            q38_http_error(error, error_capacity, "temperature must be between 0 and 2");
            goto fail;
        }
        request->has_temperature = 1;
    }
    token = q38_object_get(json, tokens, count, 0, "top_p");
    if (token >= 0) {
        if (!q38_parse_float(json, &tokens[token], &request->top_p) ||
            request->top_p <= 0.0f || request->top_p > 1.0f) {
            q38_http_error(error, error_capacity, "top_p must be in (0, 1]");
            goto fail;
        }
        request->has_top_p = 1;
    }
    token = q38_object_get(json, tokens, count, 0, "presence_penalty");
    if (token >= 0) {
        if (!q38_parse_float(json, &tokens[token], &request->presence_penalty) ||
            request->presence_penalty < 0.0f || request->presence_penalty > 2.0f) {
            q38_http_error(error, error_capacity, "presence_penalty must be between 0 and 2");
            goto fail;
        }
        request->has_presence_penalty = 1;
    }

    const int messages = q38_object_get(json, tokens, count, 0, "messages");
    if (messages < 0 || tokens[messages].type != JSMN_ARRAY) {
        q38_http_error(error, error_capacity, "messages must be a JSON array");
        goto fail;
    }
    size_t total_text = 0;
    int index = messages + 1;
    while (index < count && tokens[index].start < tokens[messages].end) {
        if (tokens[index].type != JSMN_OBJECT ||
            request->message_count >= Q38_HTTP_MAX_MESSAGES) {
            q38_http_error(error, error_capacity, "messages contains too many or invalid items");
            goto fail;
        }
        const int role = q38_object_get(json, tokens, count, index, "role");
        const int content = q38_object_get(json, tokens, count, index, "content");
        Q38HttpMessage *message = &request->messages[request->message_count];
        if (role < 0 || content < 0) {
            q38_http_error(error, error_capacity, "each message needs role and content");
            goto fail;
        }
        message->role = q38_decode_string(json, &tokens[role]);
        if (tokens[content].type == JSMN_STRING)
            message->content = q38_decode_string(json, &tokens[content]);
        else if (q38_primitive_is(json, &tokens[content], "null"))
            message->content = (char *)calloc(1u, 1u);
        if (!message->role || !message->content) {
            free(message->role);
            free(message->content);
            message->role = NULL;
            message->content = NULL;
            q38_http_error(error, error_capacity,
                           "message content must be a string or null");
            goto fail;
        }
        ++request->message_count;
        if (!q38_supported_role(message->role)) {
            q38_http_error(error, error_capacity, "unsupported message role");
            goto fail;
        }
        const int tool_call_id = q38_object_get(json, tokens, count, index,
                                                "tool_call_id");
        if (tool_call_id >= 0 &&
            !(message->tool_call_id = q38_decode_string(
                json, &tokens[tool_call_id]))) {
            q38_http_error(error, error_capacity,
                           "tool_call_id must be a string");
            goto fail;
        }
        const int calls = q38_object_get(json, tokens, count, index,
                                         "tool_calls");
        if (calls >= 0) {
            if (tokens[calls].type != JSMN_ARRAY ||
                strcmp(message->role, "assistant") != 0) {
                q38_http_error(error, error_capacity,
                               "tool_calls belongs on an assistant message");
                goto fail;
            }
            int call_index = calls + 1;
            while (call_index < count &&
                   tokens[call_index].start < tokens[calls].end) {
                if (tokens[call_index].type != JSMN_OBJECT ||
                    message->tool_call_count >= Q38_HTTP_MAX_TOOL_CALLS) {
                    q38_http_error(error, error_capacity,
                                   "too many or invalid tool_calls");
                    goto fail;
                }
                const int call_type = q38_object_get(json, tokens, count,
                                                     call_index, "type");
                const int call_id = q38_object_get(json, tokens, count,
                                                   call_index, "id");
                const int function = q38_object_get(json, tokens, count,
                                                    call_index, "function");
                if (call_type < 0 || function < 0 ||
                    !q38_token_equal(json, &tokens[call_type], "function") ||
                    tokens[function].type != JSMN_OBJECT) {
                    q38_http_error(error, error_capacity,
                                   "only function tool_calls are supported");
                    goto fail;
                }
                const int call_name = q38_object_get(json, tokens, count,
                                                     function, "name");
                const int arguments = q38_object_get(json, tokens, count,
                                                     function, "arguments");
                Q38HttpToolCall *call =
                    &message->tool_calls[message->tool_call_count];
                call->id = call_id >= 0
                         ? q38_decode_string(json, &tokens[call_id]) : NULL;
                call->name = call_name >= 0
                           ? q38_decode_string(json, &tokens[call_name]) : NULL;
                call->arguments = arguments >= 0
                                ? q38_decode_string(json, &tokens[arguments])
                                : NULL;
                ++message->tool_call_count;
                if (!call->id || !q38_tool_name_valid(call->name) ||
                    !call->arguments) {
                    q38_http_error(error, error_capacity,
                                   "invalid function tool_call");
                    goto fail;
                }
                call_index = q38_token_next(tokens, count, call_index);
            }
        }
        if (strcmp(message->role, "tool") == 0 &&
            (!message->tool_call_id || !message->content[0])) {
            q38_http_error(error, error_capacity,
                           "tool messages need tool_call_id and content");
            goto fail;
        }
        if (message->tool_call_count && !request->tool_count) {
            q38_http_error(error, error_capacity,
                           "tool_calls history requires tools");
            goto fail;
        }
        total_text += strlen(message->role) + strlen(message->content) +
                      (message->tool_call_id ? strlen(message->tool_call_id) : 0u);
        for (size_t i = 0; i < message->tool_call_count; ++i)
            total_text += strlen(message->tool_calls[i].name) +
                          strlen(message->tool_calls[i].arguments);
        if (total_text > Q38_HTTP_MAX_TEXT) {
            q38_http_error(error, error_capacity, "decoded message text is too large");
            goto fail;
        }
        index = q38_token_next(tokens, count, index);
    }
    if (request->message_count == 0) {
        q38_http_error(error, error_capacity, "messages must not be empty");
        goto fail;
    }
    free(tokens);
    return 1;

fail:
    free(tokens);
    q38_http_chat_request_free(request);
    return 0;
}

static int q38_append_system_text(Q38Buffer *system, const char *text)
{
    return (!system->length || q38_buffer_append_string(system, "\n\n")) &&
           q38_buffer_append_string(system, text);
}

static int q38_append_history_calls(Q38Buffer *prompt,
                                    const Q38HttpMessage *message,
                                    char *error, size_t error_capacity)
{
    for (size_t call_index = 0; call_index < message->tool_call_count;
         ++call_index) {
        const Q38HttpToolCall *call = &message->tool_calls[call_index];
        jsmntok_t tokens[1024];
        jsmn_parser parser;
        jsmn_init(&parser);
        const size_t length = strlen(call->arguments);
        const int count = jsmn_parse(&parser, call->arguments, length, tokens,
                                     sizeof(tokens) / sizeof(tokens[0]));
        if (count < 1 || tokens[0].type != JSMN_OBJECT ||
            tokens[0].start != 0 || tokens[0].end != (int)length) {
            q38_http_error(error, error_capacity,
                           "tool_call arguments must be one JSON object");
            return 0;
        }
        if (!q38_buffer_append_string(prompt, "<tool_call>\n<function=") ||
            !q38_buffer_append_string(prompt, call->name) ||
            !q38_buffer_append_string(prompt, ">\n")) return 0;
        int index = 1;
        while (index < count && tokens[index].start < tokens[0].end) {
            const int value = index + 1;
            if (value >= count || tokens[index].type != JSMN_STRING ||
                tokens[value].start >= tokens[0].end) {
                q38_http_error(error, error_capacity,
                               "invalid tool_call arguments object");
                return 0;
            }
            char *name = q38_decode_string(call->arguments, &tokens[index]);
            if (!q38_tool_name_valid(name)) {
                free(name);
                q38_http_error(error, error_capacity,
                               "invalid tool parameter name");
                return 0;
            }
            const int is_string = tokens[value].type == JSMN_STRING;
            char *string_value = is_string
                ? q38_decode_string(call->arguments, &tokens[value]) : NULL;
            int ok = q38_buffer_append_string(prompt, "<parameter=") &&
                     q38_buffer_append_string(prompt, name) &&
                     q38_buffer_append_string(prompt, ">");
            free(name);
            if (ok && is_string)
                ok = string_value && q38_buffer_append_string(prompt,
                                                               string_value);
            else if (ok)
                ok = q38_buffer_append(
                    prompt, call->arguments + tokens[value].start,
                    (size_t)(tokens[value].end - tokens[value].start));
            free(string_value);
            if (!ok || !q38_buffer_append_string(prompt,
                                                  "</parameter>\n"))
                return 0;
            index = q38_token_next(tokens, count, value);
        }
        if (!q38_buffer_append_string(
                prompt, "</function>\n</tool_call>\n")) return 0;
    }
    return 1;
}

static int q38_append_tools(Q38Buffer *system,
                            const Q38HttpChatRequest *request)
{
    if (!request->tool_count || request->tool_choice_none) return 1;
    static const char header[] =
        "# Tools\n\n"
        "You have access to the following functions. Their JSON schemas are "
        "listed inside <tools></tools>:\n<tools>\n";
    static const char footer[] =
        "</tools>\n\n"
        "To call a function, use exactly this XML format:\n"
        "<tool_call>\n<function=FUNCTION_NAME>\n"
        "<parameter=PARAMETER_NAME>PARAMETER_VALUE</parameter>\n"
        "</function>\n</tool_call>\n"
        "You may emit multiple <tool_call> blocks. Do not call undeclared "
        "functions or invent parameters.";
    if (system->length && !q38_buffer_append_string(system, "\n\n")) return 0;
    if (!q38_buffer_append_string(system, header)) return 0;
    for (size_t i = 0; i < request->tool_count; ++i)
        if (!q38_buffer_append_string(system, request->tools[i].function_json) ||
            !q38_buffer_append_string(system, "\n")) return 0;
    return q38_buffer_append_string(system, footer);
}

static int q38_call_id_in_message(const Q38HttpMessage *message,
                                  const char *id)
{
    for (size_t i = 0; i < message->tool_call_count; ++i)
        if (strcmp(message->tool_calls[i].id, id) == 0) return 1;
    return 0;
}

static int q38_request_has_tool(const Q38HttpChatRequest *request,
                                const char *name)
{
    for (size_t i = 0; i < request->tool_count; ++i)
        if (strcmp(request->tools[i].name, name) == 0) return 1;
    return 0;
}

char *q38_http_render_messages(const Q38HttpChatRequest *request,
                               const char *base_system, int thinking,
                               char *error, size_t error_capacity)
{
    if (!request || !request->message_count) {
        q38_http_error(error, error_capacity, "messages must not be empty");
        return NULL;
    }
    Q38Buffer system;
    Q38Buffer prompt;
    q38_buffer_init(&system);
    q38_buffer_init(&prompt);
    int ok = 1;
    if (base_system && *base_system)
        ok = q38_append_system_text(&system, base_system);
    size_t first_message = 0;
    while (first_message < request->message_count &&
           (strcmp(request->messages[first_message].role, "system") == 0 ||
            strcmp(request->messages[first_message].role, "developer") == 0)) {
        if (ok) ok = q38_append_system_text(
            &system, request->messages[first_message].content);
        ++first_message;
    }
    if (ok) ok = q38_append_tools(&system, request);
    if (ok && system.length)
        ok = q38_buffer_append_string(&prompt, "<|im_start|>system\n") &&
             q38_buffer_append(&prompt, system.data, system.length) &&
             q38_buffer_append_string(&prompt, "<|im_end|>\n");
    q38_buffer_free(&system);

    const Q38HttpMessage *tool_source = NULL;
    for (size_t i = first_message; ok && i < request->message_count; ++i) {
        const Q38HttpMessage *message = &request->messages[i];
        if (strcmp(message->role, "system") == 0 ||
            strcmp(message->role, "developer") == 0) {
            q38_http_error(error, error_capacity,
                           "system and developer messages must appear first");
            ok = 0;
        } else if (strcmp(message->role, "user") == 0) {
            tool_source = NULL;
            ok = q38_buffer_append_string(&prompt, "<|im_start|>user\n") &&
                 q38_buffer_append_string(&prompt, message->content) &&
                 q38_buffer_append_string(&prompt, "<|im_end|>\n");
        } else if (strcmp(message->role, "assistant") == 0) {
            for (size_t j = 0; j < message->tool_call_count; ++j) {
                if (!q38_request_has_tool(request,
                                          message->tool_calls[j].name)) {
                    q38_http_error(error, error_capacity,
                                   "assistant history calls an undeclared tool");
                    ok = 0;
                    break;
                }
            }
            if (!ok) break;
            tool_source = message->tool_call_count ? message : NULL;
            ok = q38_buffer_append_string(&prompt,
                                           "<|im_start|>assistant\n") &&
                 q38_buffer_append_string(&prompt, message->content);
            if (ok && message->content[0] && message->tool_call_count)
                ok = q38_buffer_append_string(&prompt, "\n");
            if (ok && message->tool_call_count)
                ok = q38_append_history_calls(&prompt, message,
                                               error, error_capacity);
            if (ok) ok = q38_buffer_append_string(&prompt, "<|im_end|>\n");
        } else if (strcmp(message->role, "tool") == 0) {
            if (!tool_source ||
                !q38_call_id_in_message(tool_source, message->tool_call_id)) {
                q38_http_error(error, error_capacity,
                               "tool result does not follow its tool_call");
                ok = 0;
                break;
            }
            if (i == first_message ||
                strcmp(request->messages[i - 1u].role, "tool") != 0)
                ok = q38_buffer_append_string(&prompt, "<|im_start|>user\n");
            if (ok) ok = q38_buffer_append_string(
                &prompt, "<tool_response>\n") &&
                q38_buffer_append_string(&prompt, message->content) &&
                q38_buffer_append_string(&prompt, "\n</tool_response>\n");
            if (ok && (i + 1u == request->message_count ||
                       strcmp(request->messages[i + 1u].role, "tool") != 0)) {
                ok = q38_buffer_append_string(&prompt, "<|im_end|>\n");
                tool_source = NULL;
            }
        }
    }
    if (ok) {
        const char *last = request->messages[request->message_count - 1u].role;
        if (strcmp(last, "user") != 0 && strcmp(last, "tool") != 0) {
            q38_http_error(error, error_capacity,
                           "the final message must have role user or tool");
            ok = 0;
        }
    }
    if (ok) ok = q38_buffer_append_string(&prompt,
                                           "<|im_start|>assistant\n");
    if (ok) ok = q38_buffer_append_string(
        &prompt, thinking ? "<think>\n" : "<think>\n\n</think>\n\n");
    if (!ok) {
        q38_buffer_free(&prompt);
        if (!error || !error[0])
            q38_http_error(error, error_capacity,
                           "out of memory rendering messages");
        return NULL;
    }
    return prompt.data;
}

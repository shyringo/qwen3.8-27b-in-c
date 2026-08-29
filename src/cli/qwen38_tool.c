#include "qwen38_tool.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define JSMN_STATIC
#include "jsmn.h"

static void set_error(char *error, size_t capacity, const char *message)
{
    if (error && capacity) snprintf(error, capacity, "%s", message);
}

void q38_tool_result_free(Q38ToolResult *result)
{
    if (!result) return;
    free(result->content);
    for (size_t i = 0; i < result->call_count; ++i) {
        free(result->calls[i].name);
        free(result->calls[i].arguments);
    }
    memset(result, 0, sizeof(*result));
}

static char *copy_range(const char *start, const char *end)
{
    if (!start || !end || end < start) return NULL;
    const size_t length = (size_t)(end - start);
    char *copy = (char *)malloc(length + 1u);
    if (!copy) return NULL;
    memcpy(copy, start, length);
    copy[length] = '\0';
    return copy;
}

static void trim(const char **start, const char **end)
{
    while (*start < *end && isspace((unsigned char)**start)) ++*start;
    while (*end > *start && isspace((unsigned char)(*end)[-1])) --*end;
}

static int valid_name(const char *start, const char *end)
{
    if (start == end || (size_t)(end - start) > 128u) return 0;
    for (const char *p = start; p < end; ++p) {
        const unsigned char c = (unsigned char)*p;
        if (!(isalnum(c) || c == '_' || c == '-' || c == '.' || c == ':'))
            return 0;
    }
    return 1;
}

static int token_next(const jsmntok_t *tokens, int count, int index)
{
    if (index < 0 || index >= count) return count;
    const int end = tokens[index].end;
    ++index;
    while (index < count && tokens[index].start < end) ++index;
    return index;
}

static int token_equal(const char *json, const jsmntok_t *token,
                       const char *text)
{
    const size_t length = strlen(text);
    return token && token->type == JSMN_STRING && token->end >= token->start &&
           (size_t)(token->end - token->start) == length &&
           memcmp(json + token->start, text, length) == 0;
}

static int object_get(const char *json, const jsmntok_t *tokens, int count,
                      int object, const char *key)
{
    if (object < 0 || object >= count || tokens[object].type != JSMN_OBJECT)
        return -1;
    int index = object + 1;
    while (index < count && tokens[index].start < tokens[object].end) {
        const int value = index + 1;
        if (value >= count || tokens[value].start >= tokens[object].end)
            return -1;
        if (token_equal(json, &tokens[index], key)) return value;
        index = token_next(tokens, count, value);
    }
    return -1;
}

static int property_type(const Q38HttpTool *tool,
                         const char *name, const char **type)
{
    *type = "string";
    if (!tool || !tool->function_json) return 1;
    jsmntok_t tokens[1024];
    jsmn_parser parser;
    jsmn_init(&parser);
    const char *json = tool->function_json;
    const int count = jsmn_parse(&parser, json, strlen(json), tokens,
                                 sizeof(tokens) / sizeof(tokens[0]));
    if (count < 1 || tokens[0].type != JSMN_OBJECT) return 0;
    const int parameters = object_get(json, tokens, count, 0, "parameters");
    const int properties = object_get(json, tokens, count, parameters,
                                      "properties");
    const int schema = object_get(json, tokens, count, properties, name);
    if (parameters < 0 || properties < 0 || schema < 0) return 0;
    const int schema_type = object_get(json, tokens, count, schema, "type");
    if (schema_type < 0) return 1;
    static const char *known[] = {
        "string", "integer", "number", "boolean", "object", "array", "null"
    };
    for (size_t i = 0; i < sizeof(known) / sizeof(known[0]); ++i) {
        if (token_equal(json, &tokens[schema_type], known[i])) {
            *type = known[i];
            return 1;
        }
    }
    return 1;
}

static int required_parameters_present(const Q38HttpTool *tool,
                                       char *const *names, size_t name_count)
{
    jsmntok_t tokens[1024];
    jsmn_parser parser;
    jsmn_init(&parser);
    const char *json = tool->function_json;
    const int count = jsmn_parse(&parser, json, strlen(json), tokens,
                                 sizeof(tokens) / sizeof(tokens[0]));
    if (count < 1 || tokens[0].type != JSMN_OBJECT) return 0;
    const int parameters = object_get(json, tokens, count, 0, "parameters");
    const int required = object_get(json, tokens, count, parameters, "required");
    if (required < 0) return 1;
    if (tokens[required].type != JSMN_ARRAY) return 0;
    int index = required + 1;
    while (index < count && tokens[index].start < tokens[required].end) {
        if (tokens[index].type != JSMN_STRING) return 0;
        int found = 0;
        for (size_t i = 0; i < name_count; ++i)
            if (token_equal(json, &tokens[index], names[i])) found = 1;
        if (!found) return 0;
        index = token_next(tokens, count, index);
    }
    return 1;
}

static const Q38HttpTool *find_tool(const Q38HttpTool *tools,
                                    size_t count, const char *name)
{
    for (size_t i = 0; i < count; ++i)
        if (tools[i].name && strcmp(tools[i].name, name) == 0) return &tools[i];
    return NULL;
}

static int valid_json_value(const char *text, size_t length,
                            const char *type)
{
    jsmntok_t tokens[512];
    jsmn_parser parser;
    jsmn_init(&parser);
    const int count = jsmn_parse(&parser, text, length, tokens,
                                 sizeof(tokens) / sizeof(tokens[0]));
    if (count < 1 || tokens[0].start != 0 || tokens[0].end != (int)length)
        return 0;
    if (strcmp(type, "object") == 0) return tokens[0].type == JSMN_OBJECT;
    if (strcmp(type, "array") == 0) return tokens[0].type == JSMN_ARRAY;
    if (tokens[0].type != JSMN_PRIMITIVE) return 0;
    if (strcmp(type, "boolean") == 0)
        return (length == 4u && memcmp(text, "true", 4u) == 0) ||
               (length == 5u && memcmp(text, "false", 5u) == 0);
    if (strcmp(type, "null") == 0)
        return length == 4u && memcmp(text, "null", 4u) == 0;
    if (strcmp(type, "integer") == 0) {
        size_t i = text[0] == '-' ? 1u : 0u;
        if (i == length) return 0;
        for (; i < length; ++i) if (!isdigit((unsigned char)text[i])) return 0;
        return 1;
    }
    if (strcmp(type, "number") == 0) {
        char *copy = copy_range(text, text + length);
        if (!copy) return 0;
        char *end = NULL;
        errno = 0;
        const double value = strtod(copy, &end);
        const int ok = !errno && end != copy && *end == '\0' && isfinite(value);
        free(copy);
        return ok;
    }
    return 0;
}

static int append_argument(Q38Buffer *json, const Q38HttpTool *tool,
                           const char *name, const char *start, const char *end)
{
    const char *type = NULL;
    if (!property_type(tool, name, &type)) return 0;
    if (!q38_buffer_append_json_string(json, name, strlen(name)) ||
        !q38_buffer_append_string(json, ":")) return 0;
    if (strcmp(type, "string") == 0)
        return q38_buffer_append_json_string(json, start, (size_t)(end - start));
    return valid_json_value(start, (size_t)(end - start), type) &&
           q38_buffer_append(json, start, (size_t)(end - start));
}

static int parse_call(const char *start, const char *end,
                      const Q38HttpTool *tools, size_t tool_count,
                      Q38ParsedToolCall *call,
                      char *error, size_t error_capacity)
{
    static const char function_prefix[] = "<function=";
    static const char function_end[] = "</function>";
    static const char parameter_prefix[] = "<parameter=";
    static const char parameter_end[] = "</parameter>";
    trim(&start, &end);
    if ((size_t)(end - start) < sizeof(function_prefix) - 1u ||
        memcmp(start, function_prefix, sizeof(function_prefix) - 1u) != 0) {
        set_error(error, error_capacity, "tool_call needs one function");
        return 0;
    }
    const char *name_start = start + sizeof(function_prefix) - 1u;
    const char *name_end = memchr(name_start, '>', (size_t)(end - name_start));
    if (!name_end || !valid_name(name_start, name_end)) {
        set_error(error, error_capacity, "invalid function name");
        return 0;
    }
    call->name = copy_range(name_start, name_end);
    const Q38HttpTool *tool = call->name
                            ? find_tool(tools, tool_count, call->name) : NULL;
    if (!call->name || !tool) {
        set_error(error, error_capacity, "model called an undeclared tool");
        return 0;
    }
    const char *function_close = strstr(name_end + 1, function_end);
    if (!function_close || function_close + sizeof(function_end) - 1u != end) {
        set_error(error, error_capacity, "incomplete or trailing function data");
        return 0;
    }
    Q38Buffer arguments;
    q38_buffer_init(&arguments);
    int ok = q38_buffer_append_string(&arguments, "{");
    size_t parameter_count = 0;
    char *parameter_names[128] = {0};
    const char *cursor = name_end + 1;
    while (ok) {
        while (cursor < function_close && isspace((unsigned char)*cursor)) ++cursor;
        if (cursor == function_close) break;
        if ((size_t)(function_close - cursor) < sizeof(parameter_prefix) - 1u ||
            memcmp(cursor, parameter_prefix, sizeof(parameter_prefix) - 1u) != 0) {
            set_error(error, error_capacity, "unexpected data inside function");
            ok = 0;
            break;
        }
        const char *param_name = cursor + sizeof(parameter_prefix) - 1u;
        const char *param_name_end = memchr(param_name, '>',
                                            (size_t)(function_close - param_name));
        if (!param_name_end || !valid_name(param_name, param_name_end)) {
            set_error(error, error_capacity, "invalid parameter name");
            ok = 0;
            break;
        }
        const char *value_start = param_name_end + 1;
        const char *parameter_close = strstr(value_start, parameter_end);
        if (!parameter_close || parameter_close > function_close) {
            set_error(error, error_capacity, "incomplete parameter");
            ok = 0;
            break;
        }
        const char *value_end = parameter_close;
        trim(&value_start, &value_end);
        char *parameter = copy_range(param_name, param_name_end);
        int duplicate = 0;
        for (size_t i = 0; parameter && i < parameter_count; ++i)
            if (strcmp(parameter_names[i], parameter) == 0) duplicate = 1;
        if (!parameter || parameter_count >= 128u || duplicate ||
            (parameter_count &&
            !q38_buffer_append_string(&arguments, ",")) ||
            !append_argument(&arguments, tool, parameter,
                             value_start, value_end)) {
            free(parameter);
            set_error(error, error_capacity, duplicate
                ? "duplicate tool parameter"
                : "parameter value does not match its schema");
            ok = 0;
            break;
        }
        parameter_names[parameter_count] = parameter;
        ++parameter_count;
        cursor = parameter_close + sizeof(parameter_end) - 1u;
    }
    if (ok && !required_parameters_present(tool, parameter_names,
                                            parameter_count)) {
        set_error(error, error_capacity, "missing required tool parameter");
        ok = 0;
    }
    if (ok) ok = q38_buffer_append_string(&arguments, "}");
    for (size_t i = 0; i < parameter_count; ++i) free(parameter_names[i]);
    if (!ok) {
        q38_buffer_free(&arguments);
        return 0;
    }
    call->arguments = arguments.data;
    return 1;
}

int q38_tool_parse(const char *text, size_t length,
                   const Q38HttpTool *tools, size_t tool_count,
                   Q38ToolResult *result,
                   char *error, size_t error_capacity)
{
    static const char tool_start[] = "<tool_call>";
    static const char tool_end[] = "</tool_call>";
    if (!text || !result) return 0;
    memset(result, 0, sizeof(*result));
    const char *begin = text;
    const char *limit = text + length;
    const char *first = strstr(begin, tool_start);
    if (!first || first >= limit) {
        result->content = copy_range(begin, limit);
        return result->content != NULL;
    }
    const char *content_end = first;
    while (content_end > begin && isspace((unsigned char)content_end[-1]))
        --content_end;
    result->content = copy_range(begin, content_end);
    if (!result->content) goto memory_fail;
    const char *cursor = first;
    while (cursor < limit) {
        while (cursor < limit && isspace((unsigned char)*cursor)) ++cursor;
        if (cursor == limit) break;
        if ((size_t)(limit - cursor) < sizeof(tool_start) - 1u ||
            memcmp(cursor, tool_start, sizeof(tool_start) - 1u) != 0) {
            set_error(error, error_capacity, "text after tool calls is not supported");
            goto fail;
        }
        if (result->call_count >= Q38_HTTP_MAX_TOOL_CALLS) {
            set_error(error, error_capacity, "too many tool calls");
            goto fail;
        }
        const char *body = cursor + sizeof(tool_start) - 1u;
        const char *close = strstr(body, tool_end);
        if (!close || close >= limit) {
            set_error(error, error_capacity, "incomplete tool_call block");
            goto fail;
        }
        Q38ParsedToolCall *call = &result->calls[result->call_count];
        if (!parse_call(body, close, tools, tool_count, call,
                        error, error_capacity)) {
            free(call->name);
            free(call->arguments);
            memset(call, 0, sizeof(*call));
            goto fail;
        }
        ++result->call_count;
        cursor = close + sizeof(tool_end) - 1u;
    }
    return 1;

memory_fail:
    set_error(error, error_capacity, "out of memory parsing tool calls");
fail:
    q38_tool_result_free(result);
    return 0;
}

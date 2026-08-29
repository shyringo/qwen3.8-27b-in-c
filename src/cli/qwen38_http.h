#ifndef QWEN38_HTTP_H
#define QWEN38_HTTP_H

#include <stddef.h>
#include <stdint.h>

#define Q38_HTTP_MAX_MESSAGES 128u
#define Q38_HTTP_MAX_TOOLS 64u
#define Q38_HTTP_MAX_TOOL_CALLS 16u
#define Q38_HTTP_MAX_BODY (1024u * 1024u)
#define Q38_HTTP_MAX_TEXT (256u * 1024u)

typedef struct {
    char *id;
    char *name;
    char *arguments;
} Q38HttpToolCall;

typedef struct {
    char *role;
    char *content;
    char *tool_call_id;
    Q38HttpToolCall tool_calls[Q38_HTTP_MAX_TOOL_CALLS];
    size_t tool_call_count;
} Q38HttpMessage;

typedef struct {
    char *name;
    char *function_json;
} Q38HttpTool;

typedef struct {
    char *model;
    Q38HttpMessage messages[Q38_HTTP_MAX_MESSAGES];
    size_t message_count;
    Q38HttpTool tools[Q38_HTTP_MAX_TOOLS];
    size_t tool_count;
    int tool_choice_none;
    uint32_t max_tokens;
    uint64_t seed;
    float temperature;
    float top_p;
    float presence_penalty;
    int stream;
    int include_usage;
    int has_max_tokens;
    int has_seed;
    int has_temperature;
    int has_top_p;
    int has_presence_penalty;
} Q38HttpChatRequest;

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
} Q38Buffer;

typedef struct {
    Q38Buffer pending;
    int (*emit)(void *context, const char *data, size_t length);
    void *context;
} Q38Utf8Stream;

void q38_http_chat_request_init(Q38HttpChatRequest *request);
void q38_http_chat_request_free(Q38HttpChatRequest *request);

int q38_http_parse_chat_request(const char *json, size_t length,
                                Q38HttpChatRequest *request,
                                char *error, size_t error_capacity);
char *q38_http_render_messages(const Q38HttpChatRequest *request,
                               const char *base_system, int thinking,
                               char *error, size_t error_capacity);

void q38_buffer_init(Q38Buffer *buffer);
void q38_buffer_free(Q38Buffer *buffer);
int q38_buffer_append(Q38Buffer *buffer, const char *data, size_t length);
int q38_buffer_append_string(Q38Buffer *buffer, const char *text);
int q38_buffer_append_json_string(Q38Buffer *buffer,
                                  const char *text, size_t length);

void q38_utf8_stream_init(Q38Utf8Stream *stream,
                          int (*emit)(void *context,
                                      const char *data, size_t length),
                          void *context);
void q38_utf8_stream_free(Q38Utf8Stream *stream);
int q38_utf8_stream_write(void *context, const char *data, size_t length);
int q38_utf8_stream_flush(Q38Utf8Stream *stream);

#endif

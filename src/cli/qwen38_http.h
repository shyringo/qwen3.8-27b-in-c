#ifndef QWEN38_HTTP_H
#define QWEN38_HTTP_H

#include <stddef.h>
#include <stdint.h>

#define Q38_HTTP_MAX_MESSAGES 128u
#define Q38_HTTP_MAX_BODY (1024u * 1024u)
#define Q38_HTTP_MAX_TEXT (256u * 1024u)

typedef struct {
    char *role;
    char *content;
} Q38HttpMessage;

typedef struct {
    char *model;
    Q38HttpMessage messages[Q38_HTTP_MAX_MESSAGES];
    size_t message_count;
    uint32_t max_tokens;
    uint64_t seed;
    float temperature;
    float top_p;
    float presence_penalty;
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

void q38_http_chat_request_init(Q38HttpChatRequest *request);
void q38_http_chat_request_free(Q38HttpChatRequest *request);

int q38_http_parse_chat_request(const char *json, size_t length,
                                Q38HttpChatRequest *request,
                                char *error, size_t error_capacity);

void q38_buffer_init(Q38Buffer *buffer);
void q38_buffer_free(Q38Buffer *buffer);
int q38_buffer_append(Q38Buffer *buffer, const char *data, size_t length);
int q38_buffer_append_string(Q38Buffer *buffer, const char *text);
int q38_buffer_append_json_string(Q38Buffer *buffer,
                                  const char *text, size_t length);

#endif

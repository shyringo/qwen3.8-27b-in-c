#ifndef QWEN38_TOOL_H
#define QWEN38_TOOL_H

#include "qwen38_http.h"

#include <stddef.h>

typedef struct {
    char *name;
    char *arguments;
} Q38ParsedToolCall;

typedef struct {
    char *content;
    Q38ParsedToolCall calls[Q38_HTTP_MAX_TOOL_CALLS];
    size_t call_count;
} Q38ToolResult;

void q38_tool_result_free(Q38ToolResult *result);
int q38_tool_parse(const char *text, size_t length,
                   const Q38HttpTool *tools, size_t tool_count,
                   Q38ToolResult *result,
                   char *error, size_t error_capacity);

#endif

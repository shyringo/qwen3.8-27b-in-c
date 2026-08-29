#include "qwen38_tool.h"

#include <stdio.h>
#include <string.h>

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        return 1; \
    } \
} while (0)

static int expect_error(const char *text, const Q38HttpTool *tools,
                        size_t count, const char *needle)
{
    Q38ToolResult result;
    char error[256] = {0};
    const int ok = q38_tool_parse(text, strlen(text), tools, count,
                                  &result, error, sizeof(error));
    if (ok) q38_tool_result_free(&result);
    return !ok && strstr(error, needle) != NULL;
}

int main(void)
{
    Q38HttpTool tools[2] = {
        {"weather", "{\"name\":\"weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}},\"required\":[\"city\"]}}"},
        {"store", "{\"name\":\"store\",\"parameters\":{\"type\":\"object\",\"properties\":{\"enabled\":{\"type\":\"boolean\"},\"items\":{\"type\":\"array\"},\"meta\":{\"type\":\"object\"}}}}"}
    };
    static const char output[] =
        "I will check.\n<tool_call><function=weather>\n"
        "<parameter=city>New York\nCity</parameter>\n"
        "<parameter=days> 3 </parameter>\n</function></tool_call>\n"
        "<tool_call><function=store>"
        "<parameter=enabled>true</parameter>"
        "<parameter=items>[1,\"two\"]</parameter>"
        "<parameter=meta>{\"x\":1}</parameter>"
        "</function></tool_call>";
    Q38ToolResult result;
    char error[256] = {0};
    const int parsed = q38_tool_parse(output, strlen(output), tools, 2,
                                      &result, error, sizeof(error));
    if (!parsed) fprintf(stderr, "parser error: %s\n", error);
    CHECK(parsed);
    CHECK(strcmp(result.content, "I will check.") == 0);
    CHECK(result.call_count == 2);
    CHECK(strcmp(result.calls[0].name, "weather") == 0);
    CHECK(strcmp(result.calls[0].arguments,
                 "{\"city\":\"New York\\nCity\",\"days\":3}") == 0);
    CHECK(strcmp(result.calls[1].arguments,
                 "{\"enabled\":true,\"items\":[1,\"two\"],\"meta\":{\"x\":1}}") == 0);
    q38_tool_result_free(&result);

    CHECK(q38_tool_parse("plain answer", 12u, tools, 2,
                          &result, error, sizeof(error)));
    CHECK(result.call_count == 0 && strcmp(result.content, "plain answer") == 0);
    q38_tool_result_free(&result);

    CHECK(expect_error("<tool_call><function=missing></function></tool_call>",
                       tools, 2, "undeclared"));
    CHECK(expect_error("<tool_call><function=weather><parameter=days>x</parameter></function></tool_call>",
                       tools, 2, "schema"));
    CHECK(expect_error("<tool_call><function=weather><parameter=days>1</parameter></function></tool_call>",
                       tools, 2, "required"));
    CHECK(expect_error("<tool_call><function=weather><parameter=country>China</parameter><parameter=city>Beijing</parameter></function></tool_call>",
                       tools, 2, "schema"));
    CHECK(expect_error("<tool_call><function=weather><parameter=city>x</parameter><parameter=city>y</parameter></function></tool_call>",
                       tools, 2, "duplicate"));
    CHECK(expect_error("<tool_call><function=store><parameter=enabled>1</parameter></function></tool_call>",
                       tools, 2, "schema"));
    CHECK(expect_error("<tool_call><function=weather>", tools, 2,
                       "incomplete"));
    CHECK(expect_error("<tool_call><function=store></function></tool_call> trailing",
                       tools, 2, "after"));
    puts("qwen38 tool-call parser tests passed");
    return 0;
}

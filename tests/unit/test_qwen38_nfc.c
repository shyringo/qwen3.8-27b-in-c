#include "tok_nfc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition, message) do { \
    if (!(condition)) { fprintf(stderr, "FAIL: %s\n", message); return 1; } \
} while (0)

static int check_normalization(const char *input, const char *expected)
{
    char *actual = NULL;
    size_t length = 0;
    if (!tok_nfc_normalize(input, strlen(input), &actual, &length)) return 0;
    const int equal = length == strlen(expected) &&
                      memcmp(actual, expected, length) == 0;
    free(actual);
    return equal;
}

int main(void)
{
    CHECK(check_normalization("A\xcc\x81", "\xc3\x81"),
          "compose Latin letter and acute accent");
    CHECK(check_normalization("\xe2\x84\xab", "\xc3\x85"),
          "canonical singleton decomposition");
    CHECK(check_normalization("q\xcc\x95\xcc\x80",
                              "q\xcc\x80\xcc\x95"),
          "canonical combining-mark order");
    CHECK(check_normalization("\xe1\x84\x92\xe1\x85\xa1\xe1\x86\xab",
                              "\xed\x95\x9c"),
          "compose Hangul jamo");
    CHECK(check_normalization("", ""), "normalize empty string");

    char invalid[] = {(char)0xc0, (char)0x80, '\0'};
    char *output = NULL;
    size_t length = 0;
    CHECK(!tok_nfc_normalize(invalid, 2, &output, &length),
          "reject malformed UTF-8");
    puts("qwen38 nfc: ok");
    return 0;
}

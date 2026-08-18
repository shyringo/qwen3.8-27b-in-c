#include "qwen38_sampler.h"

#include <stdio.h>

#define CHECK(condition, message) do { \
    if (!(condition)) { fprintf(stderr, "FAIL: %s\n", message); return 1; } \
} while (0)

int main(void)
{
    const float logits[5] = {-2.0f, 1.0f, 4.0f, 3.0f, 0.0f};
    Q38Sampler sampler;
    q38_sampler_init(&sampler, 42);
    sampler.temperature = 0.0f;
    CHECK(q38_sample(&sampler, logits, 5) == 2, "greedy sampling");
    sampler.temperature = 1.0f;
    sampler.top_k = 1;
    CHECK(q38_sample(&sampler, logits, 5) == 2, "top-k one");
    sampler.top_k = 2;
    sampler.top_p = 0.01f;
    CHECK(q38_sample(&sampler, logits, 5) == 2, "top-p retains best token");
    puts("qwen38 sampler: ok");
    return 0;
}

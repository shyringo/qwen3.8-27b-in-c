#define _POSIX_C_SOURCE 200809L

#include "qwen38_model.h"
#include "qwen38_sampler.h"
#include "qwen38_tokenizer.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

typedef struct {
    const char *model;
    const char *mtp_model;
    const char *prompt;
    const char *prompt_file;
    const char *system;
    uint32_t context;
    uint32_t max_tokens;
    uint64_t seed;
    int thinking;
    int reasoning_effort;
    int mtp;
    uint32_t mtp_depth;
    uint32_t top_k;
    float temperature;
    float top_p;
    float presence_penalty;
} Q38Options;

#define Q38_TURN_TAIL_CAPACITY 16u

typedef struct {
    uint32_t tokens[Q38_TURN_TAIL_CAPACITY];
    uint32_t count;
} Q38TurnTail;

enum {
    Q38_REASONING_LOW,
    Q38_REASONING_MEDIUM,
    Q38_REASONING_XHIGH
};

static double q38_now(void)
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec * 1e-9;
}

static void q38_usage(const char *program)
{
    fprintf(stderr,
        "usage: %s --model MODEL.gguf [--prompt TEXT] [options]\n"
        "  --prompt TEXT       run one request; omit for interactive chat\n"
        "  --prompt-file PATH  read one request from a file; use - for stdin\n"
        "  --system TEXT       system instruction for the conversation\n"
        "  --max-tokens N      maximum generated tokens (default: 1024)\n"
        "  --context N         context capacity (default: 4096)\n"
        "  --no-thinking       answer directly instead of showing reasoning\n"
        "  --reasoning-effort N  low, medium, or xhigh (default: xhigh)\n"
        "  --temperature N     0 for greedy; default: 1.0 / direct 0.7\n"
        "  --top-k N           sample from the best N tokens (default: 20)\n"
        "  --top-p N           nucleus probability; default: 0.95 / direct 0.8\n"
        "  --presence-penalty N  penalize tokens already present; default: 0 / direct 1.5\n"
        "  --mtp               enable experimental greedy MTP verification\n"
        "  --mtp-model PATH    MTP sidecar GGUF for split checkpoints\n"
        "  --mtp-depth N       MTP verification depth (default: 3)\n"
        "  --seed N            sampling seed\n", program);
}

static int q38_u32(const char *text, uint32_t *value)
{
    if (!text || !*text || text[0] == '-') return 0;
    char *end = NULL;
    errno = 0;
    const unsigned long parsed = strtoul(text, &end, 10);
    if (errno || end == text || *end || parsed > UINT32_MAX) return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static int q38_u64(const char *text, uint64_t *value)
{
    if (!text || !*text || text[0] == '-') return 0;
    char *end = NULL;
    errno = 0;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || end == text || *end) return 0;
    *value = (uint64_t)parsed;
    return 1;
}

static int q38_options(int argc, char **argv, Q38Options *options)
{
    memset(options, 0, sizeof(*options));
    options->context = 4096;
    options->max_tokens = 1024;
    options->seed = (uint64_t)time(NULL);
    options->thinking = 1;
    options->reasoning_effort = Q38_REASONING_XHIGH;
    options->mtp = 0;
    options->mtp_depth = 3;
    options->top_k = 20;
    options->temperature = 1.0f;
    options->top_p = 0.95f;
    options->presence_penalty = 0.0f;
    int temperature_set = 0;
    int top_p_set = 0;
    int presence_penalty_set = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) options->model = argv[++i];
        else if (strcmp(argv[i], "--mtp-model") == 0 && i + 1 < argc) {
            options->mtp_model = argv[++i];
            options->mtp = 1;
        }
        else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) options->prompt = argv[++i];
        else if (strcmp(argv[i], "--prompt-file") == 0 && i + 1 < argc) options->prompt_file = argv[++i];
        else if (strcmp(argv[i], "--system") == 0 && i + 1 < argc) options->system = argv[++i];
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            if (!q38_u32(argv[++i], &options->max_tokens)) return 0;
        } else if (strcmp(argv[i], "--context") == 0 && i + 1 < argc) {
            if (!q38_u32(argv[++i], &options->context)) return 0;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!q38_u64(argv[++i], &options->seed)) return 0;
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            char *end = NULL;
            const char *text = argv[++i];
            errno = 0;
            options->temperature = strtof(text, &end);
            if (errno || end == text || *end || !isfinite(options->temperature) ||
                options->temperature < 0.0f) return 0;
            temperature_set = 1;
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            if (!q38_u32(argv[++i], &options->top_k) || options->top_k == 0)
                return 0;
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            char *end = NULL;
            const char *text = argv[++i];
            errno = 0;
            options->top_p = strtof(text, &end);
            if (errno || end == text || *end || !isfinite(options->top_p) ||
                options->top_p <= 0.0f ||
                options->top_p > 1.0f) return 0;
            top_p_set = 1;
        } else if (strcmp(argv[i], "--presence-penalty") == 0 &&
                   i + 1 < argc) {
            char *end = NULL;
            const char *text = argv[++i];
            errno = 0;
            options->presence_penalty = strtof(text, &end);
            if (errno || end == text || *end ||
                !isfinite(options->presence_penalty) ||
                options->presence_penalty < 0.0f ||
                options->presence_penalty > 2.0f) return 0;
            presence_penalty_set = 1;
        } else if (strcmp(argv[i], "--reasoning-effort") == 0 && i + 1 < argc) {
            const char *effort = argv[++i];
            if (strcmp(effort, "low") == 0)
                options->reasoning_effort = Q38_REASONING_LOW;
            else if (strcmp(effort, "medium") == 0)
                options->reasoning_effort = Q38_REASONING_MEDIUM;
            else if (strcmp(effort, "xhigh") == 0)
                options->reasoning_effort = Q38_REASONING_XHIGH;
            else
                return 0;
        } else if (strcmp(argv[i], "--mtp-depth") == 0 && i + 1 < argc) {
            if (!q38_u32(argv[++i], &options->mtp_depth) ||
                options->mtp_depth == 0) return 0;
            options->mtp = 1;
        } else if (strcmp(argv[i], "--mtp") == 0) options->mtp = 1;
        else if (strcmp(argv[i], "--no-mtp") == 0) options->mtp = 0;
        else if (strcmp(argv[i], "--no-thinking") == 0) options->thinking = 0;
        else return 0;
    }
    if (!options->thinking) {
        if (!temperature_set) options->temperature = 0.7f;
        if (!top_p_set) options->top_p = 0.8f;
        if (!presence_penalty_set) options->presence_penalty = 1.5f;
    }
    if (options->mtp && !presence_penalty_set)
        options->presence_penalty = 0.0f;
    return options->model && options->context > 0 && options->max_tokens > 0 &&
           !(options->prompt && options->prompt_file);
}

static char *q38_read_prompt(const char *path)
{
    const int close_file = strcmp(path, "-") != 0;
    FILE *file = close_file ? fopen(path, "rb") : stdin;
    if (!file) {
        perror(path);
        return NULL;
    }
    size_t length = 0;
    size_t capacity = 4096;
    char *text = (char *)malloc(capacity + 1u);
    if (!text) {
        if (close_file) fclose(file);
        return NULL;
    }
    while (!feof(file)) {
        if (length == capacity) {
            if (capacity >= 64u * 1024u * 1024u) {
                fprintf(stderr, "qwen38: prompt file exceeds 64 MiB\n");
                free(text);
                if (close_file) fclose(file);
                return NULL;
            }
            capacity *= 2u;
            char *larger = (char *)realloc(text, capacity + 1u);
            if (!larger) {
                free(text);
                if (close_file) fclose(file);
                return NULL;
            }
            text = larger;
        }
        length += fread(text + length, 1, capacity - length, file);
        if (ferror(file)) {
            fprintf(stderr, "qwen38: unable to read prompt from %s\n", path);
            free(text);
            if (close_file) fclose(file);
            return NULL;
        }
    }
    if (close_file) fclose(file);
    text[length] = '\0';
    return text;
}

static int q38_emit(Q38Tokenizer *tokenizer, uint32_t token)
{
    if (q38_tokenizer_is_special(tokenizer, token)) return 1;
    char piece[1024];
    const int bytes = q38_tokenizer_decode_token(tokenizer, token,
                                                  piece, sizeof(piece));
    if (bytes < 0) return 0;
    if (bytes > 0) {
        if (fwrite(piece, 1, (size_t)bytes, stdout) != (size_t)bytes ||
            fflush(stdout) != 0) return 0;
    }
    return 1;
}

static int q38_defer_turn(Q38Tokenizer *tokenizer, Q38TurnTail *tail,
                           uint32_t pending, int include_pending,
                           int close_thinking)
{
    if (!tail) return 1;
    Q38TurnTail next = {{0}, 0};
    if (include_pending)
        next.tokens[next.count++] = pending;
    if (close_thinking) {
        static const char closure[] = "\n</think>\n\n";
        const int count = q38_tokenizer_encode(
            tokenizer, closure, sizeof(closure) - 1u,
            next.tokens + next.count,
            Q38_TURN_TAIL_CAPACITY - next.count - 1u);
        if (count <= 0 || next.count + (uint32_t)count >= Q38_TURN_TAIL_CAPACITY)
            return 0;
        next.count += (uint32_t)count;
    }
    next.tokens[next.count++] = q38_tokenizer_eos(tokenizer);
    *tail = next;
    return 1;
}

static void q38_finish_output(int closed_thinking, int truncated)
{
    if (truncated && !closed_thinking)
        fputs("\n</think>\n\n", stdout);
    else
        putchar('\n');
    fflush(stdout);
}

static void q38_report_timing(int prompt_tokens, uint32_t generated,
                              double started, double first_ready,
                              double last_ready)
{
    fprintf(stderr, "[prompt=%d tokens", prompt_tokens);
    if (generated > 0)
        fprintf(stderr, ", TTFT=%.3fs", first_ready - started);
    fprintf(stderr, ", output=%u tokens", generated);
    if (generated > 1)
        fprintf(stderr, ", TPOT=%.3fs",
                (last_ready - first_ready) / (generated - 1u));
    fprintf(stderr, "]\n");
}

static int q38_generate_mtp(Q38Model *model, Q38Tokenizer *tokenizer,
                            Q38Sampler *sampler, const Q38Options *options,
                            const float *logits, int prompt_tokens,
                            double started, int close_token,
                            Q38TurnTail *tail)
{
    const uint32_t eos = q38_tokenizer_eos(tokenizer);
    uint32_t *batch = (uint32_t *)malloc(
        options->mtp_depth * sizeof(uint32_t));
    if (!batch) return 0;
    uint32_t current = 0;
    if (!q38_sample(sampler, logits, q38_model_vocab_size(model), &current)) {
        fprintf(stderr, "qwen38: unable to sample the next token\n");
        free(batch);
        return 0;
    }
    q38_sampler_observe(sampler, current);
    uint32_t generated = 0;
    double first_ready = 0.0;
    double last_ready = 0.0;
    int closed_thinking = !options->thinking;
    int ok = 1;
    if (current != eos) {
        first_ready = last_ready = q38_now();
        ok = q38_emit(tokenizer, current);
        if (ok) {
            generated = 1;
            if ((int)current == close_token) closed_thinking = 1;
        }
    }
    while (ok && current != eos && generated < options->max_tokens) {
        uint32_t depth = options->max_tokens - generated;
        if (depth > options->mtp_depth) depth = options->mtp_depth;
        const uint32_t position = q38_model_position(model);
        const uint32_t context = q38_model_context_length(model);
        if (position >= context) {
            fprintf(stderr,
                    "qwen38: context is full; restart with a larger --context\n");
            ok = 0;
            break;
        }
        if (depth > context - position) depth = context - position;
        uint32_t count = 0;
        ok = q38_model_speculative_greedy(model, current, eos, depth,
                                           batch, &count, &logits);
        const double ready = q38_now();
        if (ok && count == 0) {
            fprintf(stderr, "qwen38: MTP verification returned no token\n");
            ok = 0;
        }
        for (uint32_t i = 0; ok && i < count; ++i) {
            current = batch[i];
            q38_sampler_observe(sampler, current);
            if (current == eos) break;
            ok = q38_emit(tokenizer, current);
            if (ok) {
                ++generated;
                last_ready = ready;
                if ((int)current == close_token) closed_thinking = 1;
            }
        }
    }
    const int truncated = ok && current != eos &&
                          generated >= options->max_tokens;
    if (ok && current == eos)
        ok = q38_defer_turn(tokenizer, tail, 0, 0, 0);
    else if (truncated)
        ok = q38_defer_turn(tokenizer, tail, current, 1,
                            options->thinking && !closed_thinking);
    free(batch);
    q38_finish_output(closed_thinking, truncated);
    if (truncated)
        fprintf(stderr,
                "qwen38: output reached --max-tokens=%u; raise the limit to continue longer\n",
                options->max_tokens);
    q38_report_timing(prompt_tokens, generated, started,
                      first_ready, last_ready);
    return ok;
}

static char *q38_prompt(const char *user, const char *system,
                        int first_turn, int thinking, int reasoning_effort)
{
    static const char xhigh_instruction[] =
        "Reasoning effort is set to xhigh. Please think carefully through the task, "
        "validate key assumptions, consider plausible alternatives, and prioritize "
        "correctness, consistency, and clarity in the final answer.";
    static const char low_instruction[] =
        "Reasoning effort is set to low. Keep your thinking brief and focused, "
        "moving directly to the conclusion without unnecessary elaboration.";
    static const char system_prefix[] = "<|im_start|>system\n";
    static const char user_prefix[] = "<|im_start|>user\n";
    static const char user_suffix[] = "<|im_end|>\n<|im_start|>assistant\n";
    static const char thinking_suffix[] = "<think>\n";
    static const char direct_suffix[] = "<think>\n\n</think>\n\n";
    const char *instruction = "";
    if (thinking && reasoning_effort == Q38_REASONING_XHIGH)
        instruction = xhigh_instruction;
    else if (thinking && reasoning_effort == Q38_REASONING_LOW)
        instruction = low_instruction;
    const int add_system = first_turn && (*instruction || (system && *system));
    const size_t system_length = system && *system ? strlen(system) : 0u;
    const size_t length = strlen(user) + strlen(user_prefix) + strlen(user_suffix) +
                          strlen(thinking ? thinking_suffix : direct_suffix) +
                          (add_system
                              ? strlen(system_prefix) +
                                strlen(instruction) +
                                (*instruction && system_length ? 2u : 0u) +
                                system_length + strlen("<|im_end|>\n") : 0u);
    char *text = (char *)malloc(length + 1u);
    if (!text) return NULL;
    text[0] = '\0';
    if (add_system) {
        strcat(text, system_prefix);
        strcat(text, instruction);
        if (*instruction && system_length) strcat(text, "\n\n");
        if (system_length) strcat(text, system);
        strcat(text, "<|im_end|>\n");
    }
    strcat(text, user_prefix);
    strcat(text, user);
    strcat(text, user_suffix);
    strcat(text, thinking ? thinking_suffix : direct_suffix);
    return text;
}

static int q38_evaluate_text(Q38Model *model, Q38Tokenizer *tokenizer,
                             Q38Sampler *sampler,
                             const char *text, const float **logits,
                             int *token_count, Q38TurnTail *tail)
{
    const size_t length = strlen(text);
    const uint32_t prefix = tail ? tail->count : 0u;
    if (length > (SIZE_MAX / sizeof(uint32_t)) - 16u - prefix) return 0;
    const size_t capacity = length + 16u + prefix;
    uint32_t *tokens = (uint32_t *)malloc(capacity * sizeof(uint32_t));
    if (!tokens) return 0;
    if (prefix) memcpy(tokens, tail->tokens, prefix * sizeof(uint32_t));
    const int encoded = q38_tokenizer_encode(tokenizer, text, length,
                                              tokens + prefix,
                                              capacity - prefix);
    const uint32_t count = encoded > 0 ? prefix + (uint32_t)encoded : 0u;
    int ok = encoded > 0;
    const uint32_t position = q38_model_position(model);
    const uint32_t context = q38_model_context_length(model);
    if (ok && count > context - position) {
        fprintf(stderr,
                "qwen38: prompt needs %u positions, but only %u remain; "
                "increase --context\n",
                count, context - position);
        ok = 0;
    }
    if (ok) ok = q38_model_prefill(model, tokens, count, logits);
    if (ok) {
        for (uint32_t i = 0; i < count; ++i)
            q38_sampler_observe(sampler, tokens[i]);
    }
    free(tokens);
    if (ok && tail) tail->count = 0;
    if (ok && token_count) *token_count = (int)count;
    return ok;
}

static int q38_generate(Q38Model *model, Q38Tokenizer *tokenizer,
                        Q38Sampler *sampler, const Q38Options *options,
                        const char *user, int first_turn, int close_token,
                        Q38TurnTail *tail)
{
    const double started = q38_now();
    char *prompt = q38_prompt(user, options->system, first_turn,
                              options->thinking, options->reasoning_effort);
    if (!prompt) return 0;
    const float *logits = NULL;
    int prompt_tokens = 0;
    const int prompt_ok = q38_evaluate_text(model, tokenizer, sampler, prompt,
                                             &logits, &prompt_tokens, tail);
    free(prompt);
    if (!prompt_ok) return 0;
    if (options->thinking) {
        fputs("<think>\n", stdout);
        fflush(stdout);
    }
    if (options->mtp && options->temperature == 0.0f)
        return q38_generate_mtp(model, tokenizer, sampler, options, logits,
                                prompt_tokens, started, close_token, tail);
    const uint32_t eos = q38_tokenizer_eos(tokenizer);
    uint32_t generated = 0;
    double first_ready = 0.0;
    double last_ready = 0.0;
    int closed_thinking = !options->thinking;
    int truncated = 0;
    int ok = 1;
    while (ok && generated < options->max_tokens) {
        uint32_t token = 0;
        if (!q38_sample(sampler, logits, q38_model_vocab_size(model),
                        &token)) {
            fprintf(stderr, "qwen38: unable to sample the next token\n");
            ok = 0;
            break;
        }
        q38_sampler_observe(sampler, token);
        const double ready = q38_now();
        if (token == eos) {
            ok = q38_defer_turn(tokenizer, tail, 0, 0, 0);
            break;
        }
        if (generated == 0) first_ready = ready;
        ok = q38_emit(tokenizer, token);
        if (!ok) break;
        ++generated;
        last_ready = ready;
        if ((int)token == close_token) closed_thinking = 1;
        if (generated == options->max_tokens) {
            truncated = 1;
            ok = q38_defer_turn(tokenizer, tail, token, 1,
                                options->thinking && !closed_thinking);
            break;
        }
        if (q38_model_position(model) >= q38_model_context_length(model)) {
            fprintf(stderr,
                    "qwen38: context is full; restart with a larger --context\n");
            ok = 0;
            break;
        }
        ok = q38_model_forward_token(model, token, &logits);
    }
    q38_finish_output(closed_thinking, truncated);
    if (truncated)
        fprintf(stderr,
                "qwen38: output reached --max-tokens=%u; raise the limit to continue longer\n",
                options->max_tokens);
    q38_report_timing(prompt_tokens, generated, started,
                      first_ready, last_ready);
    return ok;
}

int main(int argc, char **argv)
{
    if (argc == 2 && (strcmp(argv[1], "--help") == 0 ||
                      strcmp(argv[1], "-h") == 0)) {
        q38_usage(argv[0]);
        return 0;
    }
    Q38Options options;
    if (!q38_options(argc, argv, &options)) {
        q38_usage(argv[0]);
        return 2;
    }
    if (options.mtp && options.temperature != 0.0f) {
        fprintf(stderr,
                "qwen38: --mtp requires greedy decoding; add --temperature 0\n");
        return 2;
    }
    if (options.mtp && options.presence_penalty != 0.0f) {
        fprintf(stderr,
                "qwen38: --mtp requires --presence-penalty 0\n");
        return 2;
    }
#if defined(_OPENMP)
    const char *threads = getenv("QWEN38_THREADS");
    if (threads && *threads) {
        uint32_t count = 0;
        if (!q38_u32(threads, &count) || count == 0 || count > 1024) {
            fprintf(stderr,
                    "qwen38: QWEN38_THREADS must be an integer from 1 to 1024\n");
            return 2;
        }
        omp_set_num_threads((int)count);
    } else {
        const int processors = omp_get_num_procs();
        omp_set_num_threads(processors > 12 ? 12 : processors);
    }
#endif
    char *prompt_text = NULL;
    if (options.prompt_file) {
        prompt_text = q38_read_prompt(options.prompt_file);
        if (!prompt_text) return 1;
        options.prompt = prompt_text;
    }
    Q38Tokenizer *tokenizer = q38_tokenizer_open_gguf(options.model);
    Q38Model *model = tokenizer
        ? q38_model_open_gguf(options.model, options.context)
        : NULL;
    if (model && options.mtp_model &&
        !q38_model_attach_mtp_gguf(model, options.mtp_model)) {
        fprintf(stderr, "qwen38: unable to load MTP model %s\n",
                options.mtp_model);
        q38_model_close(model);
        model = NULL;
    }
    if (!tokenizer || !model) {
        q38_tokenizer_close(tokenizer);
        q38_model_close(model);
        free(prompt_text);
        return 1;
    }
    const int close_token = q38_tokenizer_find(tokenizer, "</think>");
    if (close_token < 0) {
        fprintf(stderr, "qwen38: tokenizer is missing the </think> token\n");
        q38_tokenizer_close(tokenizer);
        q38_model_close(model);
        free(prompt_text);
        return 1;
    }
    if (options.mtp && options.temperature == 0.0f &&
        !q38_model_enable_mtp(model)) {
        q38_tokenizer_close(tokenizer);
        q38_model_close(model);
        free(prompt_text);
        return 1;
    }
    const uint32_t vocabulary_size = q38_model_vocab_size(model);
    uint8_t *presence = (uint8_t *)calloc(vocabulary_size, 1u);
    if (!presence) {
        fprintf(stderr, "qwen38: unable to allocate sampler state\n");
        q38_tokenizer_close(tokenizer);
        q38_model_close(model);
        free(prompt_text);
        return 1;
    }
    Q38Sampler sampler;
    q38_sampler_init(&sampler, options.seed);
    sampler.temperature = options.temperature;
    sampler.top_k = options.top_k;
    sampler.top_p = options.top_p;
    sampler.presence_penalty = options.presence_penalty;
    sampler.presence = presence;
    sampler.presence_size = vocabulary_size;
    int ok = 1;
    if (options.prompt) {
        ok = q38_generate(model, tokenizer, &sampler, &options,
                          options.prompt, 1, close_token, NULL);
    } else {
        char *line = NULL;
        size_t capacity = 0;
        int first_turn = 1;
        Q38TurnTail tail = {{0}, 0};
        puts("Qwen3.8-27B CPU chat. Use /reset for a new chat or /exit to quit.");
        while (ok) {
            fputs("\nYou> ", stdout);
            fflush(stdout);
            ssize_t length = getline(&line, &capacity, stdin);
            if (length < 0) break;
            while (length > 0 && (line[length - 1] == '\n' ||
                                  line[length - 1] == '\r')) {
                line[--length] = '\0';
            }
            if (length == 0) continue;
            if (strcmp(line, "/quit") == 0 || strcmp(line, "/exit") == 0)
                break;
            if (strcmp(line, "/reset") == 0) {
                q38_model_reset(model);
                memset(presence, 0, vocabulary_size);
                q38_sampler_init(&sampler, options.seed);
                sampler.temperature = options.temperature;
                sampler.top_k = options.top_k;
                sampler.top_p = options.top_p;
                sampler.presence_penalty = options.presence_penalty;
                sampler.presence = presence;
                sampler.presence_size = vocabulary_size;
                first_turn = 1;
                tail.count = 0;
                puts("Conversation reset.");
                continue;
            }
            fputs("\nQwen> ", stdout);
            fflush(stdout);
            ok = q38_generate(model, tokenizer, &sampler, &options,
                               line, first_turn, close_token, &tail);
            first_turn = 0;
        }
        free(line);
    }
    free(presence);
    q38_model_close(model);
    q38_tokenizer_close(tokenizer);
    free(prompt_text);
    return ok ? 0 : 1;
}

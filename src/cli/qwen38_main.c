#define _POSIX_C_SOURCE 200809L

#include "qwen38_model.h"
#include "qwen38_http.h"
#include "qwen38_sampler.h"
#include "qwen38_tokenizer.h"

#include <errno.h>
#include <arpa/inet.h>
#include <math.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

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
    uint32_t server_port;
} Q38Options;

#define Q38_TURN_TAIL_CAPACITY 16u

typedef struct {
    uint32_t tokens[Q38_TURN_TAIL_CAPACITY];
    uint32_t count;
} Q38TurnTail;

typedef struct {
    int (*write)(void *context, const char *data, size_t length);
    void *context;
} Q38Output;

typedef struct {
    int prompt_tokens;
    uint32_t completion_tokens;
    int truncated;
} Q38GenerationStats;

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
    fprintf(stderr,
        "  --server PORT       loopback OpenAI-compatible HTTP server\n");
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
        } else if (strcmp(argv[i], "--server") == 0 && i + 1 < argc) {
            if (!q38_u32(argv[++i], &options->server_port) ||
                options->server_port == 0 || options->server_port > 65535u)
                return 0;
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
           !(options->prompt && options->prompt_file) &&
           !(options->server_port && (options->prompt || options->prompt_file));
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

static int q38_file_write(void *context, const char *data, size_t length)
{
    FILE *file = (FILE *)context;
    return fwrite(data, 1, length, file) == length && fflush(file) == 0;
}

static int q38_output_write(Q38Output *output, const char *data, size_t length)
{
    return output && output->write && output->write(output->context, data, length);
}

static int q38_output_string(Q38Output *output, const char *text)
{
    return text && q38_output_write(output, text, strlen(text));
}

static int q38_emit(Q38Tokenizer *tokenizer, uint32_t token,
                    Q38Output *output)
{
    if (q38_tokenizer_is_special(tokenizer, token)) return 1;
    char piece[1024];
    const int bytes = q38_tokenizer_decode_token(tokenizer, token,
                                                  piece, sizeof(piece));
    if (bytes < 0) return 0;
    if (bytes > 0) {
        if (!q38_output_write(output, piece, (size_t)bytes)) return 0;
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

static int q38_finish_output(int closed_thinking, int truncated,
                             Q38Output *output)
{
    if (truncated && !closed_thinking)
        return q38_output_string(output, "\n</think>\n\n");
    return q38_output_string(output, "\n");
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
                            Q38TurnTail *tail, Q38Output *output,
                            Q38GenerationStats *stats)
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
        ok = q38_emit(tokenizer, current, output);
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
            ok = q38_emit(tokenizer, current, output);
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
    if (ok) ok = q38_finish_output(closed_thinking, truncated, output);
    if (truncated)
        fprintf(stderr,
                "qwen38: output reached --max-tokens=%u; raise the limit to continue longer\n",
                options->max_tokens);
    q38_report_timing(prompt_tokens, generated, started,
                      first_ready, last_ready);
    if (stats) {
        stats->prompt_tokens = prompt_tokens;
        stats->completion_tokens = generated;
        stats->truncated = truncated;
    }
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

static int q38_generate_rendered(Q38Model *model, Q38Tokenizer *tokenizer,
                                 Q38Sampler *sampler,
                                 const Q38Options *options,
                                 const char *prompt, int close_token,
                                 Q38TurnTail *tail, Q38Output *output,
                                 Q38GenerationStats *stats)
{
    const double started = q38_now();
    const float *logits = NULL;
    int prompt_tokens = 0;
    const int prompt_ok = q38_evaluate_text(model, tokenizer, sampler, prompt,
                                             &logits, &prompt_tokens, tail);
    if (!prompt_ok) return 0;
    if (options->thinking) {
        if (!q38_output_string(output, "<think>\n")) return 0;
    }
    if (options->mtp && options->temperature == 0.0f)
        return q38_generate_mtp(model, tokenizer, sampler, options, logits,
                                prompt_tokens, started, close_token, tail,
                                output, stats);
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
        ok = q38_emit(tokenizer, token, output);
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
    if (ok) ok = q38_finish_output(closed_thinking, truncated, output);
    if (truncated)
        fprintf(stderr,
                "qwen38: output reached --max-tokens=%u; raise the limit to continue longer\n",
                options->max_tokens);
    q38_report_timing(prompt_tokens, generated, started,
                      first_ready, last_ready);
    if (stats) {
        stats->prompt_tokens = prompt_tokens;
        stats->completion_tokens = generated;
        stats->truncated = truncated;
    }
    return ok;
}

static int q38_generate(Q38Model *model, Q38Tokenizer *tokenizer,
                        Q38Sampler *sampler, const Q38Options *options,
                        const char *user, int first_turn, int close_token,
                        Q38TurnTail *tail, Q38Output *output)
{
    char *prompt = q38_prompt(user, options->system, first_turn,
                              options->thinking, options->reasoning_effort);
    if (!prompt) return 0;
    const int ok = q38_generate_rendered(model, tokenizer, sampler, options,
                                          prompt, close_token, tail, output,
                                          NULL);
    free(prompt);
    return ok;
}

typedef struct {
    char method[8];
    char path[128];
    char *body;
    size_t body_length;
} Q38IncomingRequest;

static int q38_buffer_printf(Q38Buffer *buffer, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    va_list copy;
    va_copy(copy, args);
    const int needed = vsnprintf(NULL, 0, format, copy);
    va_end(copy);
    if (needed < 0) {
        va_end(args);
        return 0;
    }
    char *text = (char *)malloc((size_t)needed + 1u);
    if (!text) {
        va_end(args);
        return 0;
    }
    vsnprintf(text, (size_t)needed + 1u, format, args);
    va_end(args);
    const int ok = q38_buffer_append(buffer, text, (size_t)needed);
    free(text);
    return ok;
}

static int q38_socket_send_all(int client, const char *data, size_t length)
{
    while (length) {
        const ssize_t sent = send(client, data, length, 0);
        if (sent < 0 && errno == EINTR) continue;
        if (sent <= 0) return 0;
        data += (size_t)sent;
        length -= (size_t)sent;
    }
    return 1;
}

static int q38_http_response(int client, int status, const char *reason,
                             const char *content_type,
                             const char *body, size_t body_length)
{
    char header[1024];
    const int length = snprintf(
        header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Headers: Authorization, Content-Type\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Connection: close\r\n\r\n",
        status, reason, content_type, body_length);
    return length > 0 && (size_t)length < sizeof(header) &&
           q38_socket_send_all(client, header, (size_t)length) &&
           q38_socket_send_all(client, body ? body : "", body_length);
}

static int q38_http_error_response(int client, int status,
                                   const char *reason, const char *message)
{
    Q38Buffer body;
    q38_buffer_init(&body);
    int ok = q38_buffer_append_string(&body, "{\"error\":{\"message\":") &&
             q38_buffer_append_json_string(&body, message, strlen(message)) &&
             q38_buffer_append_string(&body,
                 ",\"type\":\"invalid_request_error\"}}");
    if (ok) ok = q38_http_response(client, status, reason,
                                    "application/json; charset=utf-8",
                                    body.data, body.length);
    q38_buffer_free(&body);
    return ok;
}

static int q38_parse_content_length(const char *text, size_t *length)
{
    if (!text || !*text || *text == '-') return 0;
    char *end = NULL;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno || end == text || *end || value > Q38_HTTP_MAX_BODY) return 0;
    *length = (size_t)value;
    return 1;
}

static int q38_http_read_request(int client, Q38IncomingRequest *request,
                                 char *error, size_t error_capacity)
{
    enum { MAX_HEADER = 32 * 1024 };
    char header[MAX_HEADER + 1];
    size_t used = 0;
    char *separator = NULL;
    memset(request, 0, sizeof(*request));
    while (!separator) {
        if (used == MAX_HEADER) {
            snprintf(error, error_capacity, "request headers are too large");
            return 0;
        }
        const ssize_t received = recv(client, header + used, MAX_HEADER - used, 0);
        if (received < 0 && errno == EINTR) continue;
        if (received <= 0) {
            snprintf(error, error_capacity, "request ended before headers completed");
            return 0;
        }
        used += (size_t)received;
        header[used] = '\0';
        separator = strstr(header, "\r\n\r\n");
    }
    const size_t body_offset = (size_t)(separator - header) + 4u;
    *separator = '\0';
    char *line_end = strstr(header, "\r\n");
    if (!line_end) {
        snprintf(error, error_capacity, "invalid HTTP request line");
        return 0;
    }
    *line_end = '\0';
    char version[16];
    if (sscanf(header, "%7s %127s HTTP/%15s",
               request->method, request->path, version) != 3 ||
        (strcmp(version, "1.1") != 0 && strcmp(version, "1.0") != 0)) {
        snprintf(error, error_capacity, "invalid HTTP request line");
        return 0;
    }
    size_t content_length = 0;
    int saw_content_length = 0;
    char *line = line_end + 2;
    while (*line) {
        char *next = strstr(line, "\r\n");
        if (next) *next = '\0';
        if (strncasecmp(line, "Content-Length:", 15u) == 0) {
            const char *value = line + 15;
            while (*value == ' ' || *value == '\t') ++value;
            if (saw_content_length ||
                !q38_parse_content_length(value, &content_length)) {
                snprintf(error, error_capacity, "invalid Content-Length");
                return 0;
            }
            saw_content_length = 1;
        } else if (strncasecmp(line, "Transfer-Encoding:", 18u) == 0) {
            snprintf(error, error_capacity, "chunked requests are not supported");
            return 0;
        }
        if (!next) break;
        line = next + 2;
    }
    if (strcmp(request->method, "POST") == 0 && !saw_content_length) {
        snprintf(error, error_capacity, "POST requires Content-Length");
        return 0;
    }
    request->body = (char *)malloc(content_length + 1u);
    if (!request->body) {
        snprintf(error, error_capacity, "out of memory reading request");
        return 0;
    }
    size_t copied = used > body_offset ? used - body_offset : 0u;
    if (copied > content_length) copied = content_length;
    if (copied) memcpy(request->body, header + body_offset, copied);
    while (copied < content_length) {
        const ssize_t received = recv(client, request->body + copied,
                                      content_length - copied, 0);
        if (received < 0 && errno == EINTR) continue;
        if (received <= 0) {
            free(request->body);
            request->body = NULL;
            snprintf(error, error_capacity, "request body ended early");
            return 0;
        }
        copied += (size_t)received;
    }
    request->body[content_length] = '\0';
    request->body_length = content_length;
    return 1;
}

static int q38_http_buffer_write(void *context, const char *data, size_t length)
{
    return q38_buffer_append((Q38Buffer *)context, data, length);
}

static int q38_chat_append_message(Q38Buffer *prompt,
                                   const char *role, const char *content)
{
    const char *template_role = strcmp(role, "developer") == 0
                              ? "system" : role;
    return q38_buffer_append_string(prompt, "<|im_start|>") &&
           q38_buffer_append_string(prompt, template_role) &&
           q38_buffer_append_string(prompt, "\n") &&
           q38_buffer_append_string(prompt, content) &&
           q38_buffer_append_string(prompt, "<|im_end|>\n");
}

static char *q38_http_render_prompt(const Q38HttpChatRequest *request,
                                    const Q38Options *options,
                                    char *error, size_t error_capacity)
{
    if (strcmp(request->messages[request->message_count - 1u].role,
               "user") != 0) {
        snprintf(error, error_capacity, "the final message must have role user");
        return NULL;
    }
    Q38Buffer prompt;
    q38_buffer_init(&prompt);
    int ok = 1;
    if (options->system && *options->system)
        ok = q38_chat_append_message(&prompt, "system", options->system);
    for (size_t i = 0; ok && i < request->message_count; ++i)
        ok = q38_chat_append_message(&prompt, request->messages[i].role,
                                     request->messages[i].content);
    if (ok) ok = q38_buffer_append_string(&prompt, "<|im_start|>assistant\n");
    if (ok) ok = q38_buffer_append_string(
        &prompt, options->thinking ? "<think>\n" : "<think>\n\n</think>\n\n");
    if (!ok) {
        q38_buffer_free(&prompt);
        snprintf(error, error_capacity, "out of memory rendering messages");
        return NULL;
    }
    return prompt.data;
}

static void q38_sampler_reset(Q38Sampler *sampler, uint8_t *presence,
                              uint32_t vocabulary_size,
                              const Q38Options *options)
{
    memset(presence, 0, vocabulary_size);
    q38_sampler_init(sampler, options->seed);
    sampler->temperature = options->temperature;
    sampler->top_k = options->top_k;
    sampler->top_p = options->top_p;
    sampler->presence_penalty = options->presence_penalty;
    sampler->presence = presence;
    sampler->presence_size = vocabulary_size;
}

static int q38_http_chat(int client, Q38Model *model,
                          Q38Tokenizer *tokenizer, Q38Sampler *sampler,
                          uint8_t *presence, uint32_t vocabulary_size,
                          const Q38Options *base_options, int close_token,
                          const Q38IncomingRequest *incoming)
{
    static unsigned long long completion_serial = 0;
    Q38HttpChatRequest request;
    char error[256] = {0};
    if (!q38_http_parse_chat_request(incoming->body, incoming->body_length,
                                     &request, error, sizeof(error)))
        return q38_http_error_response(client, 400, "Bad Request", error);
    if (strcmp(request.model, "qwen3.8-27b-in-c") != 0 &&
        strcmp(request.model, "qwen3.8-27b") != 0) {
        q38_http_chat_request_free(&request);
        return q38_http_error_response(client, 404, "Not Found",
                                        "requested model is not available");
    }
    Q38Options options = *base_options;
    if (request.has_max_tokens) options.max_tokens = request.max_tokens;
    if (request.has_seed) options.seed = request.seed;
    if (request.has_temperature) options.temperature = request.temperature;
    if (request.has_top_p) options.top_p = request.top_p;
    if (request.has_presence_penalty)
        options.presence_penalty = request.presence_penalty;
    if (options.mtp && (options.temperature != 0.0f ||
                        options.presence_penalty != 0.0f)) {
        q38_http_chat_request_free(&request);
        return q38_http_error_response(client, 400, "Bad Request",
            "MTP requires temperature 0 and presence_penalty 0");
    }
    char *prompt = q38_http_render_prompt(&request, &options,
                                           error, sizeof(error));
    if (!prompt) {
        q38_http_chat_request_free(&request);
        return q38_http_error_response(client, 400, "Bad Request", error);
    }
    q38_model_reset(model);
    q38_sampler_reset(sampler, presence, vocabulary_size, &options);
    Q38Buffer answer;
    q38_buffer_init(&answer);
    Q38Output output = {q38_http_buffer_write, &answer};
    Q38GenerationStats stats = {0};
    const int generated = q38_generate_rendered(
        model, tokenizer, sampler, &options, prompt, close_token,
        NULL, &output, &stats);
    free(prompt);
    q38_http_chat_request_free(&request);
    if (!generated) {
        q38_buffer_free(&answer);
        return q38_http_error_response(client, 500, "Internal Server Error",
                                        "model generation failed");
    }
    while (answer.length && answer.data[answer.length - 1u] == '\n')
        answer.data[--answer.length] = '\0';
    Q38Buffer response;
    q38_buffer_init(&response);
    const time_t created = time(NULL);
    const unsigned long long serial = ++completion_serial;
    int ok = q38_buffer_printf(
        &response,
        "{\"id\":\"chatcmpl-local-%lld-%llu\","
        "\"object\":\"chat.completion\","
        "\"created\":%lld,\"model\":\"qwen3.8-27b-in-c\","
        "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
        "\"content\":",
        (long long)created, serial, (long long)created);
    if (ok) ok = q38_buffer_append_json_string(&response,
                                                answer.data ? answer.data : "",
                                                answer.length);
    if (ok) ok = q38_buffer_printf(
        &response,
        "},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,"
        "\"completion_tokens\":%u,\"total_tokens\":%u}}",
        stats.truncated ? "length" : "stop", stats.prompt_tokens,
        stats.completion_tokens,
        (unsigned)(stats.prompt_tokens + (int)stats.completion_tokens));
    q38_buffer_free(&answer);
    if (ok) ok = q38_http_response(client, 200, "OK",
                                    "application/json; charset=utf-8",
                                    response.data, response.length);
    q38_buffer_free(&response);
    return ok;
}

static int q38_http_handle(int client, Q38Model *model,
                            Q38Tokenizer *tokenizer, Q38Sampler *sampler,
                            uint8_t *presence, uint32_t vocabulary_size,
                            const Q38Options *options, int close_token)
{
    Q38IncomingRequest request;
    char error[256] = {0};
    if (!q38_http_read_request(client, &request, error, sizeof(error)))
        return q38_http_error_response(client, 400, "Bad Request", error);
    int ok = 0;
    if (strcmp(request.method, "OPTIONS") == 0) {
        ok = q38_http_response(client, 204, "No Content",
                               "application/json; charset=utf-8", "", 0u);
    } else if (strcmp(request.method, "GET") == 0 &&
               strcmp(request.path, "/health") == 0) {
        static const char body[] = "{\"status\":\"ok\"}";
        ok = q38_http_response(client, 200, "OK",
                               "application/json; charset=utf-8",
                               body, sizeof(body) - 1u);
    } else if (strcmp(request.method, "GET") == 0 &&
               strcmp(request.path, "/v1/models") == 0) {
        static const char body[] =
            "{\"object\":\"list\",\"data\":[{\"id\":\"qwen3.8-27b-in-c\","
            "\"object\":\"model\",\"owned_by\":\"local\"}]}";
        ok = q38_http_response(client, 200, "OK",
                               "application/json; charset=utf-8",
                               body, sizeof(body) - 1u);
    } else if (strcmp(request.method, "POST") == 0 &&
               strcmp(request.path, "/v1/chat/completions") == 0) {
        ok = q38_http_chat(client, model, tokenizer, sampler, presence,
                            vocabulary_size, options, close_token, &request);
    } else {
        ok = q38_http_error_response(client, 404, "Not Found",
                                      "unknown endpoint");
    }
    free(request.body);
    return ok;
}

static int q38_server(Q38Model *model, Q38Tokenizer *tokenizer,
                      Q38Sampler *sampler, uint8_t *presence,
                      uint32_t vocabulary_size, const Q38Options *options,
                      int close_token)
{
    const int server = socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) {
        perror("qwen38: socket");
        return 0;
    }
    int reuse = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons((uint16_t)options->server_port);
    if (bind(server, (const struct sockaddr *)&address, sizeof(address)) != 0 ||
        listen(server, 16) != 0) {
        perror("qwen38: bind/listen");
        close(server);
        return 0;
    }
    signal(SIGPIPE, SIG_IGN);
    fprintf(stderr,
            "qwen38: OpenAI-compatible server listening on http://127.0.0.1:%u\n",
            options->server_port);
    fprintf(stderr, "qwen38: model id qwen3.8-27b-in-c; one request at a time\n");
    for (;;) {
        const int client = accept(server, NULL, NULL);
        if (client < 0 && errno == EINTR) continue;
        if (client < 0) {
            perror("qwen38: accept");
            close(server);
            return 0;
        }
        struct timeval timeout = {30, 0};
        setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        if (!q38_http_handle(client, model, tokenizer, sampler, presence,
                              vocabulary_size, options, close_token))
            fprintf(stderr, "qwen38: request ended before a complete response\n");
        close(client);
    }
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
    Q38Output output = {q38_file_write, stdout};
    int ok = 1;
    if (options.server_port) {
        ok = q38_server(model, tokenizer, &sampler, presence,
                         vocabulary_size, &options, close_token);
    } else if (options.prompt) {
        ok = q38_generate(model, tokenizer, &sampler, &options,
                          options.prompt, 1, close_token, NULL, &output);
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
                               line, first_turn, close_token, &tail, &output);
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

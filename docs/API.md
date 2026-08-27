# Local OpenAI-compatible API

The native server keeps Qwen3.8-27B loaded so local applications can send
requests without paying model startup and mapping costs every time. It uses no
Python process or external inference runtime.

## Start the server

```bash
./qwen38.sh --server 8080 --no-thinking
```

The server listens only on `127.0.0.1`. Use this configuration in clients that
accept a custom OpenAI endpoint:

| setting | value |
|---|---|
| base URL | `http://127.0.0.1:8080/v1` |
| API key | not required; use any non-empty placeholder if a client requires one |
| model | `qwen3.8-27b-in-c` |

Stop the server with `Ctrl+C`.

## Endpoints

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

Example request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b-in-c",
    "messages": [
      {"role": "system", "content": "Answer in one concise paragraph."},
      {"role": "user", "content": "Where are the boundaries of technology?"}
    ],
    "max_tokens": 256,
    "temperature": 0
  }'
```

Without `stream`, the response is a standard `chat.completion` object with an
assistant message, finish reason, and prompt/completion/total token counts.

## Streaming

Set `stream` to `true` to receive content as server-sent events while the model
generates it:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b-in-c",
    "messages": [{"role": "user", "content": "Where are the boundaries of technology?"}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

Every JSON event is a `chat.completion.chunk` with the same completion ID and
creation time. The stream sends an assistant-role chunk, UTF-8-safe content
deltas, a final `finish_reason`, an optional usage chunk, and `data: [DONE]`.
The UTF-8 boundary buffer matters for CJK text and emoji because one model token
can end in the middle of a multibyte character.

## Supported request fields

- `model`: `qwen3.8-27b-in-c` or `qwen3.8-27b`.
- `messages`: up to 128 text messages with `system`, `developer`, `user`, or
  `assistant` roles. The final message must have the `user` role.
- `max_tokens` or `max_completion_tokens`: 1 through 65,536.
- `temperature`: 0 through 2.
- `top_p`: greater than 0 and at most 1.
- `presence_penalty`: 0 through 2.
- `seed`: a non-negative integer.
- `stream`: `true` for SSE token delivery; otherwise omitted or `false`.
- `stream_options.include_usage`: when streaming, send a final usage chunk with
  prompt, completion, and total token counts.
- `n`: omitted or `1`.

Fields omitted from a request inherit the command-line defaults used when the
server started. For example, `--no-thinking`, `--max-tokens`, and sampling
options establish server defaults while a request can override the supported
generation fields above.

## Current scope

The server is designed for a single laptop user and processes one request at a
time. Each request is stateless: clients should send the conversation history
in `messages`, as with the OpenAI Chat Completions API. The model itself stays
loaded, but recurrent and sampling state are reset before every request.

Tools, tool choice, structured output, multimodal message content,
authentication, TLS, and remote-network listening are not implemented. These
features return a clear error instead of being silently ignored. Request
bodies are limited to 1 MiB and decoded message text to 256 KiB.

# OpenRouter cache proxy

A small OpenAI-compatible Python proxy that adds explicit prompt-cache breakpoints and forwards requests to:

`https://openrouter.ai/api/v1/chat/completions`

The incoming `Authorization: Bearer ...` credential is forwarded as the OpenRouter API key. The proxy does not store keys.

## Install

```bash
cd cache-proxy
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
```

## Run

```bash
python server.py --cache-depth 2 --cache-breakpoints 4
```

The default address is `http://127.0.0.1:8787`. Available settings:

- `--host`: bind address (default `127.0.0.1`)
- `--port`: bind port (default `8787`)
- `--cache-depth`: `-1` disables markers, `0` includes all messages, and `N` excludes the last `N` messages (default `0`)
- `--cache-breakpoints`: evenly distributed explicit breakpoint count from 1 to 4 (default `4`)
- `--upstream-timeout`: OpenRouter timeout in seconds (default `180`)
- `--stats-file`: persistent counter state file (default `stats.json`)
- `--log-level`: Uvicorn log level (default `info`)

Anthropic/OpenRouter supports at most four explicit cache breakpoints. Selected string messages are converted into text blocks; selected block-array messages receive `cache_control: {"type": "ephemeral"}` on their final block.

## Docker installation

## Docker Compose

Build and start the proxy:

```bash
CACHE_DEPTH=2 CACHE_BREAKPOINTS=4 docker compose up --build -d
```

The service is available on `http://127.0.0.1:8787` by default. Compose settings can be supplied as environment variables:

- `CACHE_PROXY_PORT` (default `8787`)
- `CACHE_DEPTH` (default `0`)
- `CACHE_BREAKPOINTS` (default `4`)
- `UPSTREAM_TIMEOUT` (default `180`)
- `STATS_FILE` (default `/data/stats.json`)
- `LOG_LEVEL` (default `info`)

Stop it with:

```bash
docker compose down
```

## Request

Regular response:

```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-sonnet-4",
    "messages": [
      {"role": "system", "content": "Reusable instructions"},
      {"role": "user", "content": "Hello"}
    ]
  }'
```

Streaming response:

```bash
curl -N http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-sonnet-4",
    "stream": true,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## Models

`GET /v1/models` unauthenticated:

```bash
curl http://127.0.0.1:8787/v1/models
```

The proxy fetches OpenRouter's public model catalog once, without authorization, before the server starts accepting requests. The raw JSON response is held in memory and every `/v1/models` request receives that same snapshot. It is not refreshed while the process is running; restart the proxy to update it.

Startup fails if the initial OpenRouter model request cannot be completed successfully, ensuring a running instance always has a model catalog.

A minimal usage page is available at `GET /`. It includes persistent counters for dispatched chat requests, requests with cache reads, and cache-read tokens. The same counters are available as JSON at `GET /stats`.

Bare-metal runs store these counters in `stats.json` by default. Docker Compose stores them in `/data/stats.json` on the named `cache-proxy-data` volume, so they survive container restarts and recreation. `docker compose down -v` deletes that volume and resets the counters. A missing or invalid state file starts at zero; storage errors are logged without interrupting proxy requests. The JSON file is designed for one proxy process and should not be shared by multiple replicas.

Health check: `GET /health`.

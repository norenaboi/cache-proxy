from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
HOME_PAGE_PATH = Path(__file__).with_name("index.html")
DEFAULT_CACHE_CONTROL = {"type": "ephemeral"}
ONE_HOUR_CACHE_CONTROL = {"type": "ephemeral", "ttl": "1h"}
STATS_FIELDS = ("requests", "cache_read_requests", "cache_read_tokens")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    cache_depth: int = 0
    cache_breakpoints: int = 4
    upstream_timeout: float = 180.0
    stats_file: Path | None = None


@dataclass
class Stats:
    requests: int = 0
    cache_read_requests: int = 0
    cache_read_tokens: int = 0
    state_file: Path | None = field(default=None, repr=False)
    _lock: Lock = field(default_factory=Lock, repr=False)

    @classmethod
    def load(cls, state_file: Path | None) -> Stats:
        if state_file is None:
            return cls()

        try:
            payload = json.loads(state_file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or set(payload) != set(STATS_FIELDS):
                raise ValueError("state must contain exactly the stats counters")
            if any(
                isinstance(payload[field], bool)
                or not isinstance(payload[field], int)
                or payload[field] < 0
                for field in STATS_FIELDS
            ):
                raise ValueError("stats counters must be non-negative integers")
        except FileNotFoundError:
            return cls(state_file=state_file)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            logger.warning("Unable to load stats from %s: %s", state_file, error)
            return cls(state_file=state_file)

        return cls(
            requests=payload["requests"],
            cache_read_requests=payload["cache_read_requests"],
            cache_read_tokens=payload["cache_read_tokens"],
            state_file=state_file,
        )

    def _state(self) -> dict[str, int]:
        return {
            "requests": self.requests,
            "cache_read_requests": self.cache_read_requests,
            "cache_read_tokens": self.cache_read_tokens,
        }

    def _persist(self) -> None:
        if self.state_file is None:
            return

        temporary_path: Path | None = None
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(
                dir=self.state_file.parent,
                prefix=f".{self.state_file.name}.",
                suffix=".tmp",
            )
            temporary_path = Path(name)
            with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
                json.dump(self._state(), temporary_file, separators=(",", ":"))
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self.state_file)
        except (OSError, TypeError, ValueError) as error:
            logger.warning("Unable to persist stats to %s: %s", self.state_file, error)
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def record_request(self) -> None:
        with self._lock:
            self.requests += 1
            self._persist()

    def record_usage(self, usage: Any) -> None:
        cache_read_tokens = _cache_read_tokens(usage)
        if cache_read_tokens <= 0:
            return
        with self._lock:
            self.cache_read_requests += 1
            self.cache_read_tokens += cache_read_tokens
            self._persist()

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return self._state()


def _cache_read_tokens(usage: Any) -> int:
    if not isinstance(usage, dict):
        return 0

    details = usage.get("prompt_tokens_details")
    candidates = [
        usage.get("cache_read_input_tokens"),
        usage.get("cached_tokens"),
        details.get("cached_tokens") if isinstance(details, dict) else None,
    ]
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return 0


def _usage_from_json(content: bytes) -> Any:
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload.get("usage") if isinstance(payload, dict) else None


def _usage_from_sse_event(event: bytes) -> Any:
    for line in event.splitlines():
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            payload = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("usage"), dict):
            return payload["usage"]
    return None


def _javascript_round(value: float) -> int:
    """Match Math.round for the non-negative breakpoint calculations."""
    return math.floor(value + 0.5)


def _add_cache_control(message: Any, cache_control: dict[str, str]) -> Any:
    if not isinstance(message, dict) or not message.get("content"):
        return message

    content = message["content"]
    if isinstance(content, str):
        updated = message.copy()
        updated["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": cache_control.copy(),
            }
        ]
        return updated

    if isinstance(content, list) and content:
        final_block = content[-1]
        if not isinstance(final_block, dict):
            return message

        updated = message.copy()
        blocks = content.copy()
        blocks[-1] = {**final_block, "cache_control": cache_control.copy()}
        updated["content"] = blocks
        return updated

    return message


def apply_prompt_caching(
    messages: Any,
    cache_depth: int,
    cache_breakpoints: int,
    cache_control: dict[str, str] | None = None,
) -> Any:
    """Add evenly distributed OpenRouter cache breakpoints without mutating input."""
    if cache_depth == -1 or not isinstance(messages, list) or not messages:
        return messages

    eligible_count = max(0, len(messages) - cache_depth)
    count = min(cache_breakpoints, eligible_count)
    breakpoint_indices: set[int] = set()

    for index in range(count):
        denominator = count - 1 or 1
        selected = _javascript_round(
            (index / denominator) * (eligible_count - 1)
        )
        breakpoint_indices.add(selected)

    marker = cache_control or DEFAULT_CACHE_CONTROL
    return [
        _add_cache_control(message, marker)
        if index in breakpoint_indices
        else message
        for index, message in enumerate(messages)
    ]


def _validate_bearer(value: str | None) -> str | None:
    if not value:
        return None
    scheme, separator, token = value.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token.strip():
        return None
    return f"Bearer {token.strip()}"


def _upstream_headers(authorization: str, accept: str | None) -> dict[str, str]:
    return {
        "Authorization": authorization,
        "Content-Type": "application/json",
        "Accept": accept or "application/json",
        "User-Agent": "cache-proxy/0.1.0",
    }


def _response_headers(response: httpx.Response) -> dict[str, str]:
    headers: dict[str, str] = {}
    allowed = {
        "content-type",
        "cache-control",
        "retry-after",
        "x-request-id",
        "x-openrouter-processing-ms",
    }
    for name, value in response.headers.items():
        if name.lower() in allowed:
            headers[name] = value
    return headers


def _gateway_error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"message": message}})


def create_app(
    settings: Settings | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    configured = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(
            configured.upstream_timeout,
            connect=min(configured.upstream_timeout, 30.0),
        )
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            app.state.client = client
            app.state.stats = Stats.load(configured.stats_file)
            models_response = await client.get(
                OPENROUTER_MODELS_URL,
                headers={"Accept": "application/json"},
            )
            models_response.raise_for_status()
            app.state.models_snapshot = models_response.content
            yield

    app = FastAPI(title="OpenRouter Cache Proxy", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    @app.get("/", response_class=FileResponse)
    async def home() -> FileResponse:
        return FileResponse(HOME_PAGE_PATH, media_type="text/html")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/stats")
    async def stats(request: Request) -> dict[str, int | float]:
        return request.app.state.stats.snapshot()

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        return Response(
            content=request.app.state.models_snapshot,
            media_type="application/json",
        )

    @app.post("/v1/chat/completions")
    @app.post("/v2/chat/completions")
    async def chat_completions(request: Request) -> Response:
        authorization = _validate_bearer(request.headers.get("authorization"))
        if authorization is None:
            return _gateway_error(401, "A valid Authorization: Bearer header is required.")

        content_type = request.headers.get("content-type", "")
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            return _gateway_error(415, "Content-Type must be application/json.")

        try:
            body = json.loads(await request.body())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _gateway_error(400, "Request body must contain valid JSON.")

        if not isinstance(body, dict):
            return _gateway_error(400, "Request body must be a JSON object.")

        outgoing = deepcopy(body)
        if "messages" in outgoing:
            cache_control = (
                ONE_HOUR_CACHE_CONTROL
                if request.url.path == "/v2/chat/completions"
                else DEFAULT_CACHE_CONTROL
            )
            outgoing["messages"] = apply_prompt_caching(
                outgoing["messages"],
                configured.cache_depth,
                configured.cache_breakpoints,
                cache_control,
            )

        headers = _upstream_headers(authorization, request.headers.get("accept"))
        client: httpx.AsyncClient = request.app.state.client
        stats: Stats = request.app.state.stats
        stats.record_request()

        if body.get("stream") is True:
            upstream_request = client.build_request(
                "POST", OPENROUTER_URL, headers=headers, json=outgoing
            )
            try:
                upstream = await client.send(upstream_request, stream=True)
            except httpx.TimeoutException:
                return _gateway_error(504, "OpenRouter request timed out.")
            except httpx.RequestError:
                return _gateway_error(502, "Unable to connect to OpenRouter.")

            async def relay() -> AsyncIterator[bytes]:
                buffer = b""
                latest_usage = None
                try:
                    async for chunk in upstream.aiter_raw():
                        buffer += chunk
                        while b"\n\n" in buffer:
                            event, buffer = buffer.split(b"\n\n", 1)
                            usage = _usage_from_sse_event(event)
                            if usage is not None:
                                latest_usage = usage
                        yield chunk
                    trailing_usage = _usage_from_sse_event(buffer)
                    stats.record_usage(trailing_usage or latest_usage)
                finally:
                    await upstream.aclose()

            response_headers = _response_headers(upstream)
            response_headers["X-Accel-Buffering"] = "no"
            return StreamingResponse(
                relay(),
                status_code=upstream.status_code,
                headers=response_headers,
            )

        try:
            upstream = await client.post(OPENROUTER_URL, headers=headers, json=outgoing)
        except httpx.TimeoutException:
            return _gateway_error(504, "OpenRouter request timed out.")
        except httpx.RequestError:
            return _gateway_error(502, "Unable to connect to OpenRouter.")

        stats.record_usage(_usage_from_json(upstream.content))
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=_response_headers(upstream),
        )

    return app


def _port(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return parsed


def _cache_depth(value: str) -> int:
    parsed = int(value)
    if parsed < -1:
        raise argparse.ArgumentTypeError("must be -1 or greater")
    return parsed


def _cache_breakpoints(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 4:
        raise argparse.ArgumentTypeError("must be between 1 and 4")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add prompt-cache breakpoints and proxy chat completions to OpenRouter."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=8787)
    parser.add_argument(
        "--cache-depth",
        type=_cache_depth,
        default=0,
        help="-1 disables caching, 0 includes all messages, N excludes the last N",
    )
    parser.add_argument(
        "--cache-breakpoints",
        type=_cache_breakpoints,
        default=4,
        help="number of evenly distributed explicit breakpoints (1-4)",
    )
    parser.add_argument(
        "--upstream-timeout", type=_positive_float, default=180.0, metavar="SECONDS"
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=Path("stats.json"),
        help="persistent counter state file (default: stats.json)",
    )
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        Settings(
            cache_depth=args.cache_depth,
            cache_breakpoints=args.cache_breakpoints,
            upstream_timeout=args.upstream_timeout,
            stats_file=args.stats_file,
        )
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()

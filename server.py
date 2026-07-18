from __future__ import annotations

import argparse
import json
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
HOME_PAGE_PATH = Path(__file__).with_name("index.html")
CACHE_CONTROL = {"type": "ephemeral"}


@dataclass(frozen=True)
class Settings:
    cache_depth: int = 0
    cache_breakpoints: int = 4
    upstream_timeout: float = 180.0


def _javascript_round(value: float) -> int:
    """Match Math.round for the non-negative breakpoint calculations."""
    return math.floor(value + 0.5)


def _add_cache_control(message: Any) -> Any:
    if not isinstance(message, dict) or not message.get("content"):
        return message

    content = message["content"]
    if isinstance(content, str):
        updated = message.copy()
        updated["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": CACHE_CONTROL.copy(),
            }
        ]
        return updated

    if isinstance(content, list) and content:
        final_block = content[-1]
        if not isinstance(final_block, dict):
            return message

        updated = message.copy()
        blocks = content.copy()
        blocks[-1] = {**final_block, "cache_control": CACHE_CONTROL.copy()}
        updated["content"] = blocks
        return updated

    return message


def apply_prompt_caching(
    messages: Any, cache_depth: int, cache_breakpoints: int
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

    return [
        _add_cache_control(message) if index in breakpoint_indices else message
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
            models_response = await client.get(
                OPENROUTER_MODELS_URL,
                headers={"Accept": "application/json"},
            )
            models_response.raise_for_status()
            app.state.models_snapshot = models_response.content
            yield

    app = FastAPI(title="OpenRouter Cache Proxy", version="0.1.0", lifespan=lifespan)

    @app.get("/", response_class=FileResponse)
    async def home() -> FileResponse:
        return FileResponse(HOME_PAGE_PATH, media_type="text/html")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        return Response(
            content=request.app.state.models_snapshot,
            media_type="application/json",
        )

    @app.post("/v1/chat/completions")
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
            outgoing["messages"] = apply_prompt_caching(
                outgoing["messages"],
                configured.cache_depth,
                configured.cache_breakpoints,
            )

        headers = _upstream_headers(authorization, request.headers.get("accept"))
        client: httpx.AsyncClient = request.app.state.client

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
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
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
        )
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()

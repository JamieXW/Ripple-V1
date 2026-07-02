"""OpenTelemetry setup + request middleware (M6).

Every request gets a root span, a request ID, and an ``X-Response-Time-Ms`` header;
pipeline stages create child spans (see retrieval/pipeline.py). Spans export to the
console when ``RIPPLE_OTEL_CONSOLE=1`` — an OTLP exporter (Grafana/Jaeger/Datadog) is
a config swap away, which is the point of building on the standard.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from ripple.config import settings

_configured = False


def setup_tracing() -> None:
    """Install the tracer provider once per process."""
    global _configured
    if _configured:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": "ripple"}))
    if settings.otel_console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _configured = True


def install_middleware(app: FastAPI) -> None:
    tracer = trace.get_tracer("ripple.api")

    @app.middleware("http")
    async def trace_request(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        with tracer.start_as_current_span(
            f"{request.method} {request.url.path}",
            attributes={"ripple.request_id": request_id},
        ):
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
        return response

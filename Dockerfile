FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY server.py index.html ./

RUN pip install . \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/health', timeout=2)" || exit 1

ENTRYPOINT ["python", "server.py"]
CMD ["--host", "0.0.0.0", "--port", "8787", "--cache-depth", "0", "--cache-breakpoints", "4"]

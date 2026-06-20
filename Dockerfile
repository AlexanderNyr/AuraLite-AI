# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.11

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder-cuda
ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git build-essential curl && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml requirements.txt ./
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip wheel --wheel-dir /wheels -r requirements.txt || true
COPY . .
RUN python3 -m pip wheel --wheel-dir /wheels . || true

FROM python:${PYTHON_VERSION}-slim AS runtime-cpu
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 AURALITE_DEVICE=cpu
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install --upgrade pip && pip install -e '.[serve]' --extra-index-url https://download.pytorch.org/whl/cpu
EXPOSE 8000
HEALTHCHECK CMD python - <<'PY' || exit 1
import urllib.request
urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)
PY
CMD ["uvicorn", "server.openai_server:app", "--host", "0.0.0.0", "--port", "8000"]

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime-cuda
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip libgomp1 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN python3 -m pip install --upgrade pip && python3 -m pip install -e '.[serve,full]'
EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "server.openai_server:app", "--host", "0.0.0.0", "--port", "8000"]

# syntax=docker/dockerfile:1.7
#
# AuraLite serving images (v2.6.6).
#
#   docker build --target runtime-cpu  -t auralite-ai:cpu  .   # default in CI/releases
#   docker build --target runtime-cuda -t auralite-ai:cuda .   # docker run --gpus all …
#
# Both images:
#   * install the dependencies in a layer of their own — derived from
#     pyproject.toml by scripts/print_requirements.py, so a source edit does not
#     reinstall torch;
#   * install AuraLite itself as a regular package (not `pip install -e`);
#   * run as the unprivileged user `auralite` (uid 10001), with state, caches
#     and models under /data (AURALITE_HOME) — mount a volume there.
# Until 2.6.5 the images ran as root, used an editable install, and the CUDA
# target compiled llama.cpp on an end-of-life Python 3.10 base (it was never
# built by CI).
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AURALITE_HOME=/data
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 auralite \
    && mkdir -p /data && chown auralite:auralite /data
WORKDIR /opt/auralite
COPY pyproject.toml scripts/print_requirements.py ./

# ---------------------------------------------------------------------------
# CPU runtime: CPU-only torch wheels (small), OpenAI/Ollama-compatible server.
# ---------------------------------------------------------------------------
FROM base AS runtime-cpu
ENV AURALITE_DEVICE=cpu
RUN python print_requirements.py --pyproject pyproject.toml serve > requirements.txt \
    && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY . /tmp/src
RUN pip install --no-deps /tmp/src && rm -rf /tmp/src
USER auralite
WORKDIR /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1
CMD ["uvicorn", "server.openai_server:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------------------
# CUDA runtime: the default PyPI torch wheel bundles the CUDA libraries
# (nvidia-* wheels), so no CUDA base image is needed — only the NVIDIA driver
# and the container toolkit on the host. Adds Hugging Face + safetensors.
# GGUF on the GPU: add a prebuilt llama-cpp-python CUDA wheel in a derived image:
#   RUN pip install llama-cpp-python --only-binary llama-cpp-python \
#         --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125
# ---------------------------------------------------------------------------
FROM base AS runtime-cuda
RUN python print_requirements.py --pyproject pyproject.toml serve hf safetensors > requirements.txt \
    && pip install -r requirements.txt
COPY . /tmp/src
RUN pip install --no-deps /tmp/src && rm -rf /tmp/src
USER auralite
WORKDIR /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1
CMD ["uvicorn", "server.openai_server:app", "--host", "0.0.0.0", "--port", "8000"]

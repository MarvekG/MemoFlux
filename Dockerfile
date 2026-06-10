FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --home-dir /home/memo --shell /usr/sbin/nologin memo \
    && mkdir -p /home/memo/.memoflux/data

WORKDIR /app

ARG TORCH_VERSION=2.10.0+cpu
ARG MEMOFLUX_EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5
ARG MEMOFLUX_EMBEDDING_CACHE_DIR=/home/memo/.memoflux/data/models

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir uv \
    && UV_LINK_MODE=copy uv pip install --system --no-cache \
      --find-links https://mirrors.aliyun.com/pytorch-wheels/cpu/ \
      "torch==${TORCH_VERSION}" \
    && UV_LINK_MODE=copy uv pip install --system --no-cache -r /app/requirements.txt

RUN MEMOFLUX_EMBEDDING_MODEL="${MEMOFLUX_EMBEDDING_MODEL}" \
    MEMOFLUX_EMBEDDING_CACHE_DIR="${MEMOFLUX_EMBEDDING_CACHE_DIR}" \
    python - <<'PY'
from pathlib import Path
import os

from sentence_transformers import SentenceTransformer

cache_dir = os.environ["MEMOFLUX_EMBEDDING_CACHE_DIR"]
Path(cache_dir).mkdir(parents=True, exist_ok=True)
SentenceTransformer(os.environ["MEMOFLUX_EMBEDDING_MODEL"], cache_folder=cache_dir)
PY

COPY memoflux /app/memoflux
COPY run.py /app/run.py

RUN chown -R memo:memo /app /home/memo/.memoflux

USER memo

EXPOSE 8020

CMD ["python", "run.py"]

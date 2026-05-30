FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

ARG TORCH_VERSION=2.10.0+cpu

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir uv \
    && UV_LINK_MODE=copy uv pip install --system --no-cache \
      --find-links https://mirrors.aliyun.com/pytorch-wheels/cpu/ \
      "torch==${TORCH_VERSION}" \
    && UV_LINK_MODE=copy uv pip install --system --no-cache -r /app/requirements.txt

COPY memoflux /app/memoflux
COPY run.py /app/run.py

EXPOSE 8020

CMD ["python", "run.py"]

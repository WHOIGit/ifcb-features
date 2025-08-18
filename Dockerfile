FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

COPY . .

RUN uv pip install --system .

ENV OPENBLAS_NUM_THREADS=64 OMP_NUM_THREADS=64 NUMEXPR_NUM_THREADS=64

ENTRYPOINT ["python", "extract_slim_features.py"]

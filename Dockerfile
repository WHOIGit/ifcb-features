FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libopenblas0 libjpeg62-turbo tzdata git \
 && rm -rf /var/lib/apt/lists/*

COPY . .

RUN uv pip install --system .

ENTRYPOINT ["python", "extract_slim_features.py"]

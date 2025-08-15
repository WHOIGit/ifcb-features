FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libopenblas0 libjpeg62-turbo tzdata git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip setuptools wheel

COPY pyproject.toml ./
COPY . .

RUN pip install -v .

ENTRYPOINT ["python", "extract_slim_features.py"]

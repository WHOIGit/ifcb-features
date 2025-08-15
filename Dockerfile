FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libopenblas0 libjpeg62-turbo tzdata git \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_PROGRESS_BAR=off

COPY requirements.txt .

RUN pip install -U pip setuptools wheel \
 && pip install --prefer-binary -r requirements.txt -v

COPY pyproject.toml ./
COPY . .

RUN pip install -v .

ENTRYPOINT ["python", "extract_slim_features.py"]

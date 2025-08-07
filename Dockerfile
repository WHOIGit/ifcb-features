FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir .

ENTRYPOINT ["python", "extract_slim_features.py"]

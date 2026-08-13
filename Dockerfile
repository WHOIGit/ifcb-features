FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

COPY . .

RUN uv pip install --system .

ENV OPENBLAS_NUM_THREADS=64 OMP_NUM_THREADS=64 NUMEXPR_NUM_THREADS=64

# umask 0002 so outputs land group-writable and world-readable (0775 dirs, 0664
# files) instead of the default 0022 (0755/0644). Docker has no --umask flag, so
# it has to be set inside the container. "$@" forwards the container's arguments;
# "--" fills $0.
ENTRYPOINT ["sh", "-c", "umask 0002 && exec python extract_features_batch.py \"$@\"", "--"]

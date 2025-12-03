# Dockerfile

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# These all come with their own dependencies which is a nightmare
RUN pip install --no-cache-dir \
    "atlasopenmagic" \
    "numpy" \
    "matplotlib" \
    "uproot>=5" \
    "awkward>=2" \
    "vector" \
    "pika" \
    "requests" \
    "hist[boost]" \
    aiohttp

# Ensure no Coffea, uproot3, awkward0 can sneak in with atlas because they kill everything
RUN pip uninstall -y coffea uproot3 awkward0 || true

CMD ["python", "-u", "worker.py"]

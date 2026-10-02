# syntax=docker/dockerfile:1
#
# CPU image: works on any host (Linux, Windows, macOS/Docker Desktop, amd64 or arm64).
# For an NVIDIA GPU use docker/Dockerfile.gpu instead.
#
# Needs BuildKit (the default since Docker 23 and in Docker Desktop).
# Build from the repository root:  docker compose build
# (or: docker build -f docker/Dockerfile -t herald:cpu .)

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HERALD_PROJECT_ROOT=/app

# libsndfile: audio I/O for soundfile/librosa. libgomp: OpenMP runtime that torchaudio loads.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first. The default Linux wheels bundle the CUDA libraries (several GB).
# Keep these versions in sync with pyproject.toml: `2.8.0+cpu` satisfies its `==2.8.0` pins, so
# the install below keeps this build instead of replacing it (the check at the end enforces it).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# The dependencies come from pyproject.toml alone, so this heavy layer is rebuilt only when
# pyproject.toml changes, not on every source edit.
# monotonic-alignment-search has no wheel for every platform (e.g. linux/arm64) and compiles
# from source, so a C compiler is installed for this step only and removed again.
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install -r /tmp/requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/* /tmp/requirements.txt

# Herald itself: pure Python, so this step is quick.
COPY README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps .

# Build-time smoke test: the entry point and the native libraries load, and torch is still the
# CPU build (a CUDA torch would add several GB to the image).
RUN herald --help > /dev/null \
    && python -c "import torch, torchaudio, TTS, trainer; assert torch.version.cuda is None, torch.version.cuda"

# Run as a regular user so files written to the bind mounts are not owned by root.
# On Linux match your host user (see .env.example): HERALD_UID/HERALD_GID, then rebuild.
# Docker Desktop (macOS, Windows) maps ownership itself, so the defaults are fine there.
# --non-unique: a host GID such as 20 (macOS "staff") already exists in the image.
ARG UID=1000
ARG GID=1000
RUN groupadd --non-unique --gid "$GID" herald \
    && useradd --non-unique --uid "$UID" --gid herald --create-home herald \
    && mkdir -p dataset models runs output profiles tools \
    && chown herald:herald . dataset models runs output profiles
USER herald

# dataset/, models/, runs/, output/, profiles/ and tools/ are bind-mounted (see
# docker-compose.yml); tools/ is read-only, so it stays root-owned.
ENTRYPOINT ["herald"]
CMD ["--help"]

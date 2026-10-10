# Running Herald in Docker

Docker gives every operating system the same environment, so it is the easiest way to run Herald
on Linux and Windows without installing Python and the machine-learning libraries yourself. Native
installation (see the [README](../README.md)) is still better on a Mac: Docker Desktop on macOS has
no GPU and no sound.

## Which setup should I use?

| You have | Use | Why |
| --- | --- | --- |
| A Mac with Apple Silicon | **Native** | Docker Desktop cannot use the Mac's GPU (MPS) or play sound |
| Linux or Windows with an NVIDIA GPU | **Docker, GPU image** (or native) | Fast synthesis and training |
| Anything else | **Docker, CPU image** | Works everywhere; synthesis is slower, training very slow |

Common requirements:

- About 2 GB of disk for the base weights, plus about 2 GB per fine-tuned voice.
- [Ollama](https://ollama.com) with a model pulled (default `llama3.2:3b`) for `chat`. Docker
  Compose starts one for you.
- Native only: Python 3.11 (3.12 is allowed by the package metadata but untested).
- Training runs on CUDA when there is one, otherwise on the CPU: the trainer has no Apple MPS
  support, so a Mac trains on the CPU and needs a lot of patience.

## Step by step

`docker-compose.yml` runs an `ollama` service (the official image) and a `herald` service. The
dataset, models, runs and output directories are bind-mounted from the host, so the big files
never enter the image.

```bash
cp .env.example .env                                  # optional: read the comments in it
mkdir -p dataset models runs output                   # otherwise Docker creates them as root
docker compose build
docker compose run --rm herald download-checkpoints   # base weights into ./models/xtts_v2
docker compose run --rm herald synthesize "Hello." --reference-wav dataset/my_voice.wav -o output/hello.wav
docker compose run --rm herald chat --no-play          # text only; needs the ollama service (see below)
```

Only `chat` needs Ollama:

```bash
docker compose up -d --wait ollama
docker compose exec ollama ollama pull llama3.2:3b
```

- **CPU image** (`docker/Dockerfile`, the default): `python:3.11-slim` with CPU-only torch;
  amd64 and arm64.
- **NVIDIA GPU image** (`docker/Dockerfile.gpu`): Linux amd64 with the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
  and a driver that supports CUDA 12.8. It speeds up training as well as synthesis. Add the GPU
  file to every command, or set `COMPOSE_FILE` in `.env` (see `.env.example`):

  ```bash
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm herald train
  ```

- **Files owned by the wrong user (Linux).** Files written to the bind mounts belong to the
  container user (UID/GID 1000). Match your own by setting `HERALD_UID` and `HERALD_GID` in
  `.env` (`id -u`, `id -g`), then `docker compose build`.
- **Use an Ollama that already runs on the host** instead of the container (recommended on a
  Mac, where containers get no GPU):

  ```bash
  HERALD_OLLAMA_URL=http://host.docker.internal:11434 docker compose run --rm herald chat --no-play
  ```

  In Windows PowerShell set the variable first: `$env:HERALD_OLLAMA_URL = "http://host.docker.internal:11434"`,
  then run `docker compose run --rm herald chat --no-play`. Or put the line in the `.env` file.

- Ollama's port is not published, because a native Ollama usually owns 11434. Add
  `ports: ["11434:11434"]` to the `ollama` service to reach it from the host (it has no
  authentication).

> **macOS note.** Docker Desktop runs containers in a Linux VM with **no GPU passthrough and no
> audio device**. Synthesis there is CPU-only, and training is CPU-only in a VM with limited
> memory. For training and for speed, run natively. `herald chat` in a container cannot play
> sound: add `--no-play --save-dir output/chat` and open the WAV files it saves there, or run
> `chat` natively (it can still talk to the Ollama container).

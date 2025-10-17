# BioFoundation + Evo2 Docker Setup

Docker environment for running biofoundation with evo2 support. Uses NVIDIA's PyTorch base image with all CUDA/cuDNN dependencies pre-configured.

## Requirements

- Docker with NVIDIA Container Toolkit
- GPU with compute capability >= 8.9 (Ada/Hopper) for full FP8 support

## Setup

### Configure NVIDIA Runtime (if not already done)

If Docker doesn't recognize the NVIDIA runtime, configure it:

```bash
# Configure Docker to use NVIDIA runtime (nvidia-ctk should already be installed)
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify nvidia runtime is available
sudo docker info | grep -i runtime
# Should show: Runtimes: io.containerd.runc.v2 nvidia runc
```

### Fix Docker Permissions

**Option 1: Add user to docker group (recommended)**

```bash
sudo usermod -aG docker $USER
newgrp docker  # Apply group change immediately, or log out and back in
```

**Option 2: Use sudo with docker commands**

If you prefer not to add yourself to the docker group, run all docker commands with `sudo`:

```bash
sudo docker compose build local
sudo docker compose run --rm local bash
```

### Install NVIDIA Container Toolkit (if not already installed)

If `nvidia-ctk` command is not found, install the toolkit:

```bash
# Add NVIDIA's GPG key and repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install NVIDIA Container Toolkit
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Quick Start

```bash
# From this directory (docker/evo2/)

# Local development (includes your local changes)
./run.sh local

# Or using GitHub version
./run.sh github
```

## Manual Build

```bash
# Local development
docker compose build local
docker compose run --rm local bash

# GitHub install
docker compose build github
docker compose run --rm github bash
```

## Usage Inside Container

Both biofoundation and evo2 are available:

```python
# Use biofoundation
import biofoundation

# Use evo2
from evo2 import Evo2
model = Evo2('evo2_7b')
```

## Files

- `Dockerfile` - Local development build
- `Dockerfile.github` - GitHub install build
- `docker-compose.yml` - Docker Compose configuration
- `run.sh` - Helper script for building and running

#!/bin/bash
# Helper script to build and run Docker container with Evo2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VARIANT=${1:-local}

if [ "$VARIANT" = "github" ]; then
    echo "Building Docker image with GitHub install..."
    docker compose build github
    echo "Starting container..."
    docker compose run --rm github bash
else
    echo "Building Docker image with local install..."
    docker compose build local
    echo "Starting container..."
    docker compose run --rm local bash
fi

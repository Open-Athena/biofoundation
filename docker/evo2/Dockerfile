# Dockerfile for BioFoundation + Evo2 (local development)
# Uses NVIDIA's PyTorch image with all CUDA/cuDNN dependencies pre-installed

FROM nvcr.io/nvidia/pytorch:25.04-py3

RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-tomli \
    && rm -rf /var/lib/apt/lists/*

# Install evo2 (handles complex dependencies)
RUN pip install evo2

# Copy and install biofoundation from repository root
COPY ../.. /workspace/biofoundation
WORKDIR /workspace/biofoundation

# Install biofoundation with legacy resolver to handle constraints
RUN pip install -e . --use-deprecated=legacy-resolver --break-system-packages

# Verify installations
RUN python -c "import biofoundation; print('✓ BioFoundation installed')"
RUN python -c "from evo2 import Evo2; print('✓ Evo2 installed')"

WORKDIR /workspace

CMD ["bash"]

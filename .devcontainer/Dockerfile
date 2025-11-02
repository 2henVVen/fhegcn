FROM mcr.microsoft.com/devcontainers/base:noble
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    sudo \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install openfhe --break-system-packages
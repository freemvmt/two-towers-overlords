#!/usr/bin/env bash
# run like `source setup.sh` on any remote to ensure active shell is set up with venv

# ensure we have all the utils we need
apt-get update
apt-get install -y vim rsync git nvtop htop tmux curl ca-certificates git-lfs
apt-get upgrade -y

# load env vars into shell
cp .env.example .env
set -a
source .env
set +a

# install uv (and setup custom cache dirs if we have runpod storage)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
if [[ -n "$SSH_CONNECTION" && -d /workspace ]]; then
  echo "🐧 Running on remote runpod with storage attached - setting custom uv/hf cache dir"
  mkdir -p /workspace/.cache/uv
  mkdir -p /workspace/.cache/datasets_cache
  export UV_CACHE_DIR="/workspace/.cache/uv"
  export HF_DATASETS_CACHE="/workspace/.cache/datasets_cache"
fi

# install python packages (using nightly index for latest torch if beast mode enabled)
if [[ "$BEAST_MODE" == "1" ]]; then
  echo "🔥 BEAST_MODE enabled - using nightly config for torch prereleases"
  uv sync --prerelease=allow --config-file uv.nightly.toml
else
  uv sync
fi

# activate virtual environment for running python scripts
source .venv/bin/activate

# finally, if GET_DOCKER env var is set, install it
if [[ "$GET_DOCKER" == "1" ]]; then
  echo "🐳 GET_DOCKER enabled - installing Docker for the server"
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null
  apt-get update
  apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

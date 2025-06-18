#!/usr/bin/env bash
# run like `source setup.sh` (on remote) to ensure active shell is set up with venv

# ensure we have all the utils we need
apt update
apt install -y vim rsync git nvtop htop tmux curl git-lfs
apt upgrade -y

# load env vars into shell
cp .env.example .env
set -a
source .env
set +a

# install uv and sync (using custom cache dir if we have runpod storage)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
if [[ -n "$SSH_CONNECTION" && -d /workspace ]]; then
  echo "🐧 Running on remote runpod with storage attached - setting custom uv cache dir"
  mkdir -p /workspace/.cache/uv
  export UV_CACHE_DIR=/workspace/.cache/uv
fi
uv sync

# activate virtual environment for running python scripts
source .venv/bin/activate

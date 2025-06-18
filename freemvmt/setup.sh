#!/usr/bin/env bash
# run like `source setup.sh` (on remote) to ensure active shell is set up with venv
BEAST_MODE=$BEAST_MODE

# ensure we have all the utils we need
apt update
apt install -y vim rsync git nvtop htop tmux curl git-lfs
apt upgrade -y

# load env vars into shell
cp .env.example .env
set -a
source .env
set +a

# install uv (and setup custom cache dir if we have runpod storage)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
if [[ -n "$SSH_CONNECTION" && -d /workspace ]]; then
  echo "🐧 Running on remote runpod with storage attached - setting custom uv cache dir"
  mkdir -p /workspace/.cache/uv
  export UV_CACHE_DIR=/workspace/.cache/uv
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

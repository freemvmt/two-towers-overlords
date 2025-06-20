#!/usr/bin/env bash
# run like `source setup.sh` on any remote to ensure active shell is set up with venv

# ensure we have all the utils we need
apt-get update
apt-get install -y vim rsync git nvtop htop tmux curl ca-certificates git-lfs lsof
apt-get upgrade -y

# get env vars sent via send script and load into shell
cp ~/.env .env
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

# cd into backend dir, where the action happens
cd backend

# install python packages (using nightly index for latest torch if beast mode enabled)
if [[ "$BEAST_MODE" == "1" ]]; then
  echo "🔥 BEAST_MODE enabled - using nightly config for torch prereleases"
  uv sync --prerelease=allow --config-file uv.nightly.toml
else
  uv sync
fi

# if RUN_REDIS env var is set, run a redis server in a tmux session on the remote
if [[ "$RUN_REDIS" == "1" ]]; then
  echo "🚀 RUN_REDIS is set - installing and starting Redis server in the background"
  apt-get install -y redis-server
  # ensure any existing redis server is stopped before starting it (check with `lsof -i :6379`)
  pkill redis-server 2>/dev/null || true
  cp redis.runpod.conf /tmp/redis.conf
  sleep 2
  # run redis in a tmux session so we can attach to see logs / stop it (kill any existing session first)
  tmux kill-session -t redis 2>/dev/null || true
  tmux new-session -d -s redis 'redis-server /tmp/redis.conf'
  # give Redis a moment to start and check if it's running
  sleep 3
  if tmux ls | grep -q redis; then
    echo "✅ Redis server started in tmux session 'redis'"
  else
    echo "❌ Failed to start Redis server, try running: `redis-server /tmpredis.conf --daemonize yes`"
  fi
fi

# finally, activate virtual environment for running python scripts
source .venv/bin/activate

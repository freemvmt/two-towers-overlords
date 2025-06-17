#!/usr/bin/env bash
# run like `source setup.sh` to ensure active shell is set up with venv
apt update
# ensure we have all the utils we need
apt install -y vim rsync git nvtop htop tmux curl git-lfs
# apt upgrade -y
# install uv and sync
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync
# activate virtual environment for running python scripts
source .venv/bin/activate

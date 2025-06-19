#!/usr/bin/env bash
# run like `./ssh.sh` on tmp runpod, after sending your private key and this script

# start ssh agent, add key, go to /workspace
eval "$(ssh-agent -s)"
ssh-add .ssh/id_ed25519
cd /workspace

# ensure we have git, clone repo, cd in
apt-get update && apt-get install -y git
git clone git@github.com:freemvmt/two-towers-overlords.git
cd two-towers-overlords

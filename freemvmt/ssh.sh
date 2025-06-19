#!/usr/bin/env bash
# run like `./ssh.sh` on tmp runpod, after sending your private key and this script
# will set up ssh agent and pull down repo into /workspace dir

eval "$(ssh-agent -s)"
ssh-add .ssh/id_ed25519
cd /workspace
git clone git@github.com:freemvmt/two-towers-overlords.git
cd two-towers-overlords

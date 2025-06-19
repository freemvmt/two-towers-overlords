#!/usr/bin/env bash
# run like `./ssh.sh` (on tmp remote, after sending private key) to set up ssh agent and pull down repo

eval "$(ssh-agent -s)"
ssh-add .ssh/id_ed25519
git clone git@github.com:freemvmt/two-towers-overlords.git

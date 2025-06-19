#!/usr/bin/env bash
# run like `./send.sh` on local, to prepare remote to run ssh.sh

if [[ -z "${1-}" ]]; then
    REMOTE="mlx"
else
    REMOTE="$1"
fi

scp ~/.ssh/id_ed25519 "$REMOTE:~/.ssh/id_ed25519"
scp ssh.sh "$REMOTE:ssh.sh"

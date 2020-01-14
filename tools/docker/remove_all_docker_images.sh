#!/usr/bin/env bash
set -ux
docker image ls -aq | xargs docker rmi --force
docker system prune --force
docker image ls -aq | xargs docker rmi --force

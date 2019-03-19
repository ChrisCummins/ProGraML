#!/usr/bin/env bash
#
# Run PhD build environment with useful directories mounted.
set -eux

# Create the mapped directory so that docker doesn't (as root).
mkdir -pv $HOME/.cache/bazel/_bazel_docker

docker run \
  -v/var/run/docker.sock:/var/run/docker.sock \
  -v$HOME/.cache/bazel/_bazel_docker:$HOME/.cache/bazel/_bazel_docker \
  -v$PHD:/phd \
  -it phd_build $@

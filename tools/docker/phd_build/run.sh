#!/usr/bin/env bash
#
# Run PhD build environment with useful directories mounted.
#
# The configuration is set so that a persistent bazel cache located at
# ~/.cache/bazel/_bazel_docker is mapped into the container, speeding up repeat
# bazel builds.
#
# Usage:
#
#   ./tools/docker/phd_build/run.sh
#       Drop into interactive bash session.
#
#   ./tools/docker/phd_build/run.sh <command...>
#       Run the specified command.
#
#   ENTRYPOINT=foo ./tools/docker/phd_build/run.sh <args...>
#       Use the specified entrypoint and run with args.
set -eu

# Create the mapped directory so that docker doesn't (as root).
mkdir -pv $HOME/.cache/bazel/_bazel_docker

set +u
if [[ -z $ENTRYPOINT ]]; then
  prearg="-it"
else
  prearg="--entrypoint $ENTRYPOINT"
fi

set -x
docker run \
  -v/var/run/docker.sock:/var/run/docker.sock \
  -v$HOME/.cache/bazel/_bazel_docker:/home/docker/.cache/bazel/_bazel_docker \
  -v$PHD:/phd \
  $prearg chriscummins/phd_build:latest $@

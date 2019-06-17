#!/usr/bin/env bash
#
# Run PhD base environment.
#
# Usage:
#
#   ./tools/docker/phd_base/run.sh
#       Drop into interactive bash session.
#
#   ./tools/docker/phd_base/run.sh <command...>
#       Run the specified command.
#
#   ENTRYPOINT=foo ./tools/docker/phd_base/run.sh <args...>
#       Use the specified entrypoint and run with args.
set -eu

set +u
if [[ -z $ENTRYPOINT ]]; then
  prearg="-it"
else
  prearg="--entrypoint $ENTRYPOINT"
fi

set -x
docker run \
  -v/var/run/docker.sock:/var/run/docker.sock \
  $prearg phd_base $@

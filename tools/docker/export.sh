#!/usr/bin/env bash
#
# Build and publish a docker image.
#
# Usage:
#
#     # bazel run //tools/docker:export ~/path/to/dir/containing/dockerfile

# --- begin labm8 init ---
f=phd/labm8/sh/app.sh
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  { echo>&2 "ERROR: cannot find $f"; exit 1; }; f=
# --- begin app init ---

set -eu

main() {
  local path="$1"

  local image="$(basename $path)"

  local version="$(cat $(DataPath phd/version.txt))"

  docker build -t chriscummins/"$image":latest "$path"
  docker build -t chriscummins/"$image":"$version" "$path"
  docker push chriscummins/"$image":latest
  docker push chriscummins/"$image":"$version"
  docker rmi chriscummins/"$image":latest
  docker rmi chriscummins/"$image":"$version"
}
main "$@"

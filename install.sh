#!/usr/bin/env bash
help() {
  cat <<EOF
Install the command line ProGraML tools.

Usage:

  $ bazel run -c opt //:install [prefix]

Installs the command line tools to [prefix]/bin. [prefix] defaults to ~/.local/opt/programl.
EOF
}

# --- begin runfiles.bash initialization v2 ---
# Copy-pasted from the Bazel Bash runfiles library v2.
set -uo pipefail; f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
 source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
 source "$0.runfiles/$f" 2>/dev/null || \
 source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
 source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
 { echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v2 ---

set -euo pipefail

PACKAGE="$(rlocation programl/package.tar.bz2)"

main() {
  set +u
  if [[ "$1" == "--help" ]]; then
    help
    exit 1
  fi
  set -u

  local prefix=${1:-~/.local/opt/programl}
  mkdir -p "$prefix"

  echo "Installing ProGraML to $prefix"
  echo
  tar xjvf "$PACKAGE" -C "$prefix" | sed 's/^/    /'
  echo
  echo "===================================================="
  echo "ProGraML is now installed."
  echo "Add the following to your ~/.$(basename $SHELL)rc:"
  echo
  echo "export PATH=$prefix/bin:\$PATH"
  echo "export LD_LIBRARY_PATH=$prefix/lib:\$LD_LIBRARY_PATH"
}
main "$@"

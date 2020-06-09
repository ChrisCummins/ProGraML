#!/usr/bin/env bash
help() {
  cat <<EOF
Install the command line ProGraML tools.

Usage:

  $ install [prefix]

Installs the command line tools to [prefix]/bin. [prefix] defaults to ~/.local/opt/programl.
EOF
}

# --- begin labm8 init ---
f=phd/labm8/sh/app.sh
# shellcheck disable=SC1090
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null ||
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null ||
  source "$0.runfiles/$f" 2>/dev/null ||
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null ||
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null ||
  {
    echo >&2 "ERROR: cannot find $f"
    exit 1
  }
f=
# --- end app init ---

set -euo pipefail

BINARIES=(
  "$(DataPath phd/programl/cmd/analyze)"
  "$(DataPath phd/programl/cmd/clang2graph)"
  "$(DataPath phd/programl/cmd/graph2cdfg)"
  "$(DataPath phd/programl/cmd/graph2dot)"
  "$(DataPath phd/programl/cmd/graph2json)"
  "$(DataPath phd/programl/cmd/llvm2graph)"
  "$(DataPath phd/programl/cmd/pbq)"
  "$(DataPath phd/programl/cmd/xla2graph)"
)

if [[ $(uname) == Darwin ]]; then
  LLVM_LIBS="$(DataPath llvm_mac/lib)"
else
  LLVM_LIBS="$(DataPath llvm_linux/lib)"
fi

main() {
  set +u
  if [[ "$1" == "--help" ]]; then
    help
    exit 1
  fi
  set -u

  local prefix=${1:-~/.local/opt/programl}
  mkdir -p "$prefix/bin" "$prefix/lib"

  echo "Installing ProGraML command line tools ..."
  echo
  for bin in "${BINARIES[@]}"; do
    dst="$prefix/bin/$(basename $bin)"
    echo "    $dst"
    rm -f "$dst"
    cp $bin "$dst"
  done

  echo
  echo "Installing libraries ..."
  rsync -ah "$LLVM_LIBS/" "$prefix/lib"

  echo
  echo "===================================================="
  echo "To use them, add the following to your ~/.$(basename $SHELL)rc:"
  echo
  echo "export PATH=$prefix/bin:\$PATH"
  echo "export LD_LIBRARY_PATH=$prefix/lib:\$LD_LIBRARY_PATH"
}
main "$@"

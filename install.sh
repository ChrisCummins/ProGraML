#!/usr/bin/env bash
help() {
  cat <<EOF
Install the command line ProGraML tools.

Usage:

  $ bazel run -c opt //:install [prefix]

Installs the command line tools to [prefix]/bin. [prefix] defaults to ~/.local/opt/programl.
EOF
}

# --- begin labm8 init ---
f=programl/external/labm8/labm8/sh/app.sh
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
  "$(DataPath programl/bin/analyze)"
  "$(DataPath programl/bin/clang2graph-10)"
  "$(DataPath programl/bin/graph2cdfg)"
  "$(DataPath programl/bin/graph2dot)"
  "$(DataPath programl/bin/graph2json)"
  "$(DataPath programl/bin/llvm2graph-10)"
  "$(DataPath programl/bin/pbq)"
  "$(DataPath programl/bin/xla2graph)"
)

if [[ $(uname) == Darwin ]]; then
  LLVM_LIBS="$(DataPath clang-llvm-10.0.0-x86_64-apple-darwin/lib)"
else
  LLVM_LIBS="$(DataPath clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/lib)"
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

  # Create symlinks from $bin-$vesion versioned binaries to $bin.
  rm -f "$prefix/bin/clang2graph" "$prefix/bin/llvm2graph"
  echo "    $prefix/bin/clang2graph"
  ln -s clang2graph-10 "$prefix/bin/clang2graph"
  echo "    $prefix/bin/llvm2graph"
  ln -s llvm2graph-10 "$prefix/bin/llvm2graph"

  echo
  echo "Installing libraries to $prefix/libs ..."
  rsync -ah --delete --exclude '*.a' "$LLVM_LIBS/" "$prefix/lib/"

  # NOTE(github.com/ChrisCummins/ProGraML/issues/134): Workaround for load-time
  # errors when systems expect LLVMPolly.so to have the "lib" name prefix.
  if [[ -f "$prefix/lib/LLVMPolly.so" ]]; then
    ln -s "$prefix/lib/LLVMPolly.so" "$prefix/lib/libLLVMPolly.so"
  fi

  echo
  echo "===================================================="
  echo "To use them, add the following to your ~/.$(basename $SHELL)rc:"
  echo
  echo "export PATH=$prefix/bin:\$PATH"
  echo "export LD_LIBRARY_PATH=$prefix/lib:\$LD_LIBRARY_PATH"
}
main "$@"

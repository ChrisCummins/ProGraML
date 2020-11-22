#!/usr/bin/env bash

help() {
  cat <<EOF
unpack_ir_text: Extract text IR files from IR protocol buffers.

Usage:

  unpack_ir_text <dataset_path>

WHere <dataset_path> is the root directory of the dataflow dataset.

For each file in <dataset_path>/ir, this creates a corresponding entry
in <dataset_path>/ll.
EOF
}

# --- begin labm8 init ---
f=programl/labm8/sh/app.sh
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

set -eu

# Paths to binaries.
PBQ="$(DataPath programl/bin/pbq)"
if [[ -f /usr/local/opt/findutils/libexec/gnubin/find ]]; then
  FIND=/usr/local/opt/findutils/libexec/gnubin/find
else
  FIND=find
fi

# Unpack binary IR protocol buffers to text files.
#
# This must be run from the root of the dataset directory.
unpack_ir_text() {
  echo "Unpacking IR files ..."
  mkdir -p ll
  # One big GNU parallel invocation to enumerate all IRs and feed them through the
  # pbq command to extract the Ir.text field.
  "$FIND" ir -type f -printf '%f\n' |
    sed 's/\.Ir\.pb$//' |
    parallel \
      "$PBQ" Ir.text --stdin_fmt=pb \
      '<' ir/{}.Ir.pb \
      '>' ll/{}.ll
}

main() {
  set +u
  if [[ -z "$1" ]]; then
    help >&2
    exit 1
  fi
  set -u

  if [[ "$1" == "--help" ]]; then
    help
    exit 1
  fi

  if [[ ! -d "$1" ]]; then
    echo "Directory not found: $1" >&2
    exit 1
  fi

  cd "$1"
  unpack_ir_text
}
main $@

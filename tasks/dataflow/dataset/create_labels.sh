#!/usr/bin/env bash

help() {
  cat <<EOF
create_labels: Generate dataflow analysis labels from.

Usage:

  create_labels <dataset_path> [analysis ...]

WHere <dataset_path> is the root directory of the dataflow dataset,
and [analysis ...] is an optional list of analyses to generate labels
for. If not provided, all analyses are run.

For each analyses that is run, graph files from <dataset_path>/graphs
are read and the results of the analysis are written to
<dataset_path>/labels/<analysis>.
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
ANALYZE="$(DataPath programl/programl/bin/analyze)"
if [[ -f /usr/local/opt/findutils/libexec/gnubin/find ]]; then
  FIND=/usr/local/opt/findutils/libexec/gnubin/find
else
  FIND=find
fi

# Generate labels for all graphs using an analysis.
#
# This must be run from the root of the dataset directory.
#
# This assumes that the `analyze` command is in $PATH.
run_analysis() {
  local analysis="$1"

  echo "Generating $analysis labels ..."
  mkdir -p labels/"$analysis"
  # Enumerate all program graphs and feed them through the `analyze` command.
  "$FIND" graphs -type f -printf '%f\n' |
    sed 's/\.ProgramGraph\.pb$//' |
    parallel timeout -s9 60 "$ANALYZE" "$analysis" \
      '<' graphs/{}.ProgramGraph.pb --stdin_fmt=pb \
      '>' labels/"$analysis"/{}.ProgramGraphFeaturesList.pb --stdout_fmt=pb
  # Any failed analyses will appear as empty files.
  "$FIND" labels/"$analysis" -type f -size 0 -delete
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
  shift

  if [ $# -eq 0 ]; then
    run_analysis reachability
    run_analysis dominance
    run_analysis datadep
    run_analysis liveness
    run_analysis subexpressions
  else
    for arg in "$@"; do
      run_analysis $arg
    done
  fi
}
main $@

#!/usr/bin/env bash

help() {
  cat <<EOF
count_instances: Count the number of input/output graph pairs for each analysis.

Usage:

  count_instances <dataset_path> [analysis ...]

Where <dataset_path> is the root directory of the dataflow dataset,
and [analysis ...] is an optional list of analyses to generate labels
for. If not provided, all analyses are run.

For each analyses that is run, a list of graph counts, one for
each for label features list, is produced and written to
<dataset_path>/labels/<analysis>_graph_counts.csv.

The output is in comma-separated format with a header row. The first column is
the basename and the second column is number of graphs in that file. Sum the
total graph counts across all analyses using:

  $ awk -F"," '{total+=$2} END {print total}' labels/*_graph_counts.csv
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

set -euo pipefail

# Paths to binaries.
PBQ="$(DataPath programl/bin/pbq)"
if [[ -f /usr/local/opt/findutils/libexec/gnubin/find ]]; then
  FIND=/usr/local/opt/findutils/libexec/gnubin/find
else
  FIND=find
fi

# Count the number of node labels for the given analysis.
#
# This must be run from the root of the dataset directory.
count_instances() {
  local analysis="$1"
  local outfile="labels/${analysis}_graph_counts.csv"

  echo "Counting $analysis instances ..."
  rm -f "$outfile"

  echo "graph_name,graph_count" >"$outfile"
  # A "find -exec" loop to run pbq on each file and print its basename and node count to CSV format.
  "$FIND" labels/"$analysis" \
    -type f \
    -exec sh -c \
    "$PBQ"' "SELECT COUNT(graph) FROM ProgramGraphFeaturesList" --stdin_fmt=pb < "{}" | sed "s/^/$(basename {} .ProgramGraphFeaturesList.pb),/"' \; \
    >>"$outfile"
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
    count_instances reachability
    count_instances dominance
    count_instances datadep
    count_instances liveness
    count_instances subexpressions
  else
    for arg in "$@"; do
      count_instances $arg
    done
  fi
}
main $@

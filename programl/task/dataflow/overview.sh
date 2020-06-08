#!/usr/bin/env bash

help() {
  cat <<EOF
overview: Print an overview of dataset directory.

Usage:

  overview <dataset_path>

Where <dataset_path> is the root directory of the dataflow dataset.
EOF
}

set -euo pipefail

if [[ -f /usr/local/opt/findutils/libexec/gnubin/find ]]; then
  FIND=/usr/local/opt/findutils/libexec/gnubin/find
else
  FIND=find
fi

# Overview of dataset stats.
# Must be called from the root of the dataflow dataset.
overview() {
  echo "=========================="
  echo "#. runs: " $("$FIND" -L logs -mindepth 3 -maxdepth 3 -type d | wc -l)
  echo
  echo "Epochs: "
  for run in $("$FIND" -L logs -mindepth 3 -maxdepth 3 -type d | sort); do
    echo "    $("$FIND" $run/epochs -type f | wc -l) $run"
  done

  echo
  echo "=========================="
  echo "Dataset job logs:"
  "$FIND" labels -name '*.txt' -mindepth 1 -maxdepth 1 -type f | sort | xargs wc -l | sed 's/^/    /'

  echo
  echo "=========================="
  echo "Directory sizes:"
  for dir in graphs $("$FIND" labels -mindepth 1 -maxdepth 1 -type d | sort); do
    echo "    $("$FIND" $dir -type f | wc -l) $dir"
  done

  echo
  echo "=========================="
  echo "Splits:"
  for dir in train val test; do
    echo "    $("$FIND" $dir -type l | wc -l) $dir"
  done

  echo
  echo "=========================="
  echo "Graph sources:"
  "$FIND" graphs -type f -printf "%f\n" >graphs_list.txt
  grep -v poj104 graphs_list.txt | grep -v github | cut -d'.' -f1 | sort | uniq -c
  grep '^poj104' graphs_list.txt | cut -d'_' -f1 | sort | uniq -c
  grep '^github' graphs_list.txt | cut -d'.' -f1,3 | sort | uniq -c

  if [[ -d ll ]]; then
    echo
    echo "=========================="
    echo "IR line counts:"
    echo "BLAS          $( ($FIND ll -type f -name 'blas*.ll' -print0 | xargs -0 cat) | wc -l)"
    echo "Github-C      $( ($FIND ll -type f -name 'github.*.c.ll' -print0 | xargs -0 cat) | wc -l)"
    echo "Github-OpenCL $( ($FIND ll -type f -name 'github.*.cl.ll' -print0 | xargs -0 cat) | wc -l)"
    echo "Github-Swift  $( ($FIND ll -type f -name 'github.*.swift.ll' -print0 | xargs -0 cat) | wc -l)"
    echo "Linux         $( ($FIND ll -type f -name 'linux*.ll' -print0 | xargs -0 cat) | wc -l)"
    echo "NPB           $( ($FIND ll -type f -name 'npb-*.ll' -print0 | xargs -0 cat) | wc -l)"
    echo "OpenCL        $( ($FIND ll -type f -name 'opencl*.ll' -print0 | xargs -0 cat) | wc -l)"
    echo "OpenCV        $( ($FIND ll -type f -name 'opencv*.ll' -print0 | xargs -0 cat) | wc -l)"
    echo "POJ-104       $( ($FIND ll -type f -name 'poj104*.ll' -print0 | xargs -0 cat) | wc -l)"
    echo "TensorFlow    $( ($FIND ll -type f -name 'tensorflow*.ll' -print0 | xargs -0 cat) | wc -l)"
    echo
    echo "=========================="
    echo "IR file counts:"
    echo "BLAS          $($FIND ll -type f -name 'blas*.ll' | wc -l)"
    echo "Github-C      $($FIND ll -type f -name 'github.*.c.ll' | wc -l)"
    echo "Github-OpenCL $($FIND ll -type f -name 'github.*.cl.ll' | wc -l)"
    echo "Github-Swift  $($FIND ll -type f -name 'github.*.swift.ll' | wc -l)"
    echo "Linux         $($FIND ll -type f -name 'linux*.ll' | wc -l)"
    echo "Linux         $($FIND ll -type f -name 'npb-*.ll' | wc -l)"
    echo "OpenCL        $($FIND ll -type f -name 'opencl*.ll' | wc -l)"
    echo "OpenCV        $($FIND ll -type f -name 'opencv*.ll' | wc -l)"
    echo "POJ-104       $($FIND ll -type f -name 'poj104*.ll' | wc -l)"
    echo "TensorFlow    $($FIND ll -type f -name 'tensorflow*.ll' | wc -l)"
  fi
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
  set +u
  if [[ -n "$1" ]]; then
    help >&2
    exit 1
  fi
  set -u

  overview
}
main $@

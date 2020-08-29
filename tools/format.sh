#!/usr/bin/env bash
#
# Run the project code formatters.
set -eu

if [[ ! -f WORKSPACE ]]; then
    echo "Must be ran from project root!" >&2
fi

find . -name '*.h' | xargs clang-format -i
find . -name '*.cc' | xargs clang-format -i

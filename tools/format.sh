#!/usr/bin/env bash
#
# Run the project code formatters.
set -eu

if [[ ! -f WORKSPACE ]]; then
    echo "Must be ran from project root!" >&2
fi

if which nproc &>/dev/null ; then
    nproc="$(nproc)"
else
    nproc=1
fi

find . \( -name 'BUILD' -o -name 'WORKSPACE' \) -print0 | xargs -0 buildifier
find . -name '*.h' -print0 | xargs -0 -n8 -P"$nproc" clang-format -i
find . -name '*.cc' -print0 | xargs -0 -n8 -P"$nproc" clang-format -i
find . -name '*.py' -print0 | xargs -0 -n8 -P"$nproc" isort --profile black
find . -name '*.py' -print0 | xargs -0 -n8 -P"$nproc" black --quiet --target-version py36

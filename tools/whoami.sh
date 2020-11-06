#!/usr/bin/env bash
#
# Print diagnostic information describing the system and some key dependencies.
# This script serves two purposes:
#    1) Sanity-check that the minimum bazel environment is working.
#    2) Provide version information useful for helping reproduce an issue.
#
# Usage
# =====
#
# If bazel does not work, you can run it directly:
#     $ ./tools/whoami.sh
#
# When reporting issues, please provide the full output of this script.

indent() {
  $@ | sed 's/^/    /'
}

git_status() {
  if [[ -d .git ]]; then
    echo "Git"
    echo "==="
    echo "    Revision: $(git rev-parse HEAD 2>/dev/null)"
    echo "    Upstream: $(git config --get remote.origin.url 2>/dev/null)"
    indent git status
  fi
}

host_status() {
  if [[ -f /etc/os-release ]]; then
    echo
    echo "Linux Host"
    echo "=========="
    indent cat /etc/os-release
  fi

  if [[ -f /usr/bin/sw_vers ]]; then
    echo
    echo "macOS Host"
    echo "=========="
    indent /usr/bin/sw_vers
  fi
}

bazel_status() {
  if which bazel &>/dev/null; then
    echo
    echo "Bazel"
    echo "====="
    indent bazel version 2>/dev/null
  fi
}

cxx_status() {
  if which c++ &>/dev/null; then
    echo
    echo "C++"
    echo "==="
    indent which c++
    indent c++ --version
  fi
}

python_status() {
  local python=python3
  if which $python &>/dev/null; then
    echo
    echo "Python"
    echo "======"
    echo "    Path: $(which $python 2>&1)"
    echo "    Version: $($python --version 2>&1)"
    echo
    echo "Pip packages"
    echo "============"
    indent $python -m pip freeze
  fi
}

main() {
  git_status
  host_status
  bazel_status
  cxx_status
  python_status
}
main

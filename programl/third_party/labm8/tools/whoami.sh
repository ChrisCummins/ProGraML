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
# Ideally, run this script using bazel:
#     $ bazel run //tools:whoami
#
# If bazel does not work, you can run it directly:
#     $ ./tools/whoami.sh
#
# When reporting issues, please provide the full output of this script.
echo "tools/workspace_status.txt"
echo "=========================="
cat tools/workspace_status.txt

if [[ -f /etc/os-release ]]; then
  echo
  echo "/etc/os-release"
  echo "==============="
  cat /etc/os-release
fi

if which python; then
  echo
  echo "Python"
  echo "======"
  echo "Path:    $(which python 2>&1)"
  echo "Version: $(python --version 2>&1)"
  echo "$(python -m pip freeze)"
fi

if which python3; then
  echo
  echo "Python 3"
  echo "========"
  echo "Path:    $(which python3 2>&1)"
  echo "Version: $(python3 --version 2>&1)"
  echo "$(python3 -m pip freeze)"
fi

#!/usr/bin/env bash
#
# This script exports the environment variables which are used for bazel
# invocations. Use it as a wrapper around a bazel invocation, optionally
# using `exec -i` to produce a clean environment:
#
#     $ exec -i tools/bazel_env.sh bazel build //...
#
# This script assumes that /bin, /usr/bin, and /usr/local/bin contain the
# necessary build tools, which is true for a standard Linux / macOS
# environment. Please report bugs and issues to:
#   <https://github.com/ChrisCummins/phd/issues>
set -eu

# Accepts an array of directories and returns a colon separated path
# of all of the directories that exist, in order.  Example usage:
#
#    dirs=("/usr/local/bin" /usr/bin "/not a real path")
#    unset FOO
#    FOO=$(build_path "${dirs[@]}")
#    echo $FOO
#    # Outputs: /usr/local/bin:/usr/bin
build_path() {
  local _dir=""
  local _path=""

  for _dir in "$@"; do
    if [ -d $_dir ]; then
      _path=$_path:$_dir
    fi
  done

  _path=${_path:1}
  echo $_path
}

# Build our path.
path_dirs=(
  /usr/local/opt/llvm/bin
  /usr/local/opt/gnu-sed/libexec/gnubin
  /Library/TeX/Distributions/.DefaultTeX/Contents/Programs/texbin
  /usr/bin
  /usr/local/bin
  /bin
)
export PATH="$(build_path ${path_dirs[@]})"

# Increase the timeout on docker image pulls from the default 600s.
# See: https://github.com/bazelbuild/rules_docker
export PULLER_TIMEOUT=3600

$@

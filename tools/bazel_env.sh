#!/usr/bin/env bash
#
# This script exports the environment variables which are used for bazel
# invocations. Use it as a wrapper around a bazel invocation, optionally
# using `exec -i` to produce a clean environment:
#
#     $ exec -i tools/bazel_env.sh bazel build //...
#
# This script assumes that /bin, /usr/bin, and /usr/local/bin contain the
# necessary build tools, which is true for the standard Linux / macOS
# environments I use. Your mileage may vary.
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

# Note(github.com/ChrisCummins/phd/issues/55): On macOS, custom LDFLAGS and
# CPPFLAGS are required to pip build MySQLdb:
if [[ -d /usr/local/opt/openssl ]]; then
  export LDFLAGS="-L/usr/local/opt/openssl/lib"
  export CPPFLAGS="-I/usr/local/opt/openssl/include"
fi

# Increase the timeout on docker image pulls from the default 600s.
# See: https://github.com/bazelbuild/rules_docker
export PULLER_TIMEOUT=3600

if [[ -f "/usr/local/opt/llvm/bin/clang" ]]; then
  export CC=/usr/local/opt/llvm/bin/clang
  export CXX=/usr/local/opt/llvm/bin/clang++
fi

$@

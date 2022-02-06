#!/usr/bin/env bash
#
# Generate key-value pairs for bazel workspace build status.
# See: https://docs.bazel.build/versions/master/user-manual.html#flag--workspace_status_command
#
# The output of this script is consumed by //:make_build_info_pbtxt to generate
# the build info proto.
set -eu

# Volatile keys.
echo "SECONDS_SINCE_EPOCH $(date +%s)"
echo "RANDOM_HASH $(cat /dev/urandom | head -c16 | md5sum 2>/dev/null | cut -f1 -d' ')"

# The path of the workspace. This leaks the out-of-tree build abstraction
# enforced by bazel, and this directory is likely not available from within
# bazel test sandbox. Access this value only when truly required, such as
# from within tool that modify or export the source tree.
echo "STABLE_UNSAFE_WORKSPACE" $(pwd)

echo "STABLE_VERSION" $(cat version.txt)

echo "STABLE_GIT_COMMIT $(git rev-parse HEAD)"
echo "STABLE_GIT_URL $(git ls-remote --get-url $(git rev-parse --abbrev-ref @{upstream} 2>/dev/null | cut -f1 -d'/') 2>/dev/null || echo null)"

if [[ -z $(git status -s) ]]; then
  echo "STABLE_GIT_DIRTY false"
else
  echo "STABLE_GIT_DIRTY true"
fi

# Linux version.
if [[ "$OSTYPE" == "linux-gnu" ]]; then
  echo "STABLE_ARCH linux_amd64"
elif [[ "$OSTYPE" == "darwin"* ]]; then
  echo "STABLE_ARCH darwin_amd64"
else
  echo "STABLE_ARCH $OSTYPE"
fi

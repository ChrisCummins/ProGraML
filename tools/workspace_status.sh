#!/usr/bin/env bash
#
# Generate key-value pairs for bazel workspace build status.
# See: https://docs.bazel.build/versions/master/user-manual.html#flag--workspace_status_command
set -eu

# Volatile keys.
echo "SECONDS_SINCE_EPOCH $(date +%s)"
echo "RANDOM_HASH $(cat /dev/urandom | head -c16 | md5sum 2>/dev/null | cut -f1 -d' ')"

# Stable keys.
echo "STABLE_HOST $(hostname)"
echo "STABLE_USER $(id -un)"

echo "STABLE_GIT_COMMIT_HASH $(git rev-parse HEAD)"
echo "STABLE_GIT_COMMIT_TIMESTAMP $(git show -s --format=%ct HEAD)"
echo "STABLE_GIT_COMMIT_AUTHOR $(git show -s --format='%cn <%ce>' HEAD)"

if [[ -z $(git status -s) ]]; then
  echo "STABLE_GIT_REPO_DIRTY false"
else
  echo "STABLE_GIT_REPO_DIRTY true"
fi

echo "STABLE_GIT_BRANCH $(git branch | grep \* | cut -d ' ' -f2)"
echo "STABLE_GIT_REMOTE $(git rev-parse --abbrev-ref --symbolic-full-name @{u})"
echo "STABLE_GIT_REMOTE_URL $(git ls-remote --get-url $(git rev-parse --abbrev-ref @{upstream} | cut -f1 -d'/'))"

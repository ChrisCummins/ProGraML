#!/usr/bin/env bash
#
# Generate key-value pairs for bazel workspace build status.
# See: https://docs.bazel.build/versions/master/user-manual.html#flag--workspace_status_command

# Volatile keys.
echo "BUILD_TIMESTAMP $(date +%s)"
echo "BUILD_TIMESTAMP_NATURAL $(date)"

# Stable keys.
echo "STABLE_BUILD_GIT_HASH $(git rev-parse HEAD)"
echo "STABLE_BUILD_HOST $(hostname)"
echo "STABLE_BUILD_USER $USER"

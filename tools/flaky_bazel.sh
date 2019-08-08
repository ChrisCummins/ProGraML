#!/usr/bin/env bash
#
# Bazel builds are flaky on Travis CI, often caused by network timeouts due to
# slow connections. To reduce the number of restarted builds caused by
# false-positive errors, this script runs the given arguments for $max_attempts
# before failing.
#
# Example usage:
#    $ ./tools/flaky_bazel.sh test //...
# <to be reverted>
set -eu
export max_attempts=5
export returncode=1
for _ in $(seq $max_attempts); do bazel $@ && { returncode=0; break; } || echo "> returncode: $?"; done
exit $returncode

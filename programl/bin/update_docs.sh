#!/usr/bin/env bash
#
# Regenerate the documentation files in //Documentation/bin.
#
#     $ programl/bin/update_docs.sh
#
# This script must be run from the root of the bazel workspace.
#
# This builds all of the documentation targets defined in //Documentation/bin, and then
# generates a markdown index of the files at //Documentation/CommandLineTools.md.
#
# Run this script any time you make a change to the command line tools usage and commit the changes.
#
set -euo pipefail

main() {
  local workspace="$(pwd)"
  rm -f Documentation/bin/*.txt
  bazel build //Documentation/bin:all
  cp -v bazel-bin/Documentation/bin/* Documentation/bin

  local index=Documentation/CommandLineTools.md
  echo -e "# Command-line Tools\n" >$index
  for doc in Documentation/bin/*.txt; do
    # absl flags for python binaries includes the full runfiles prefix in the module name, e.g.
    #
    # /private/var/tmp/_bazel_cec/a28f2c41c8a7559baae3041c83801353/sandbox/darwin-sandbox/627/execroot/programl/bazel-out/host/bin/bin/inst2vec.runfiles/programl/bin/inst2vec.py:
    #     --dataset: The path of a directory to process. When set, this changes ...
    #
    # This hacky sed invocation strips the runfiles component, leaving:
    #
    # bin/inst2vec.py:
    #     --dataset: The path of a directory to process. When set, this changes ...
    sed -r 's,/.+\.runfiles/[^/]+/,,' -i $doc
    # Create an entry in the index of command-line documentation files.
    echo -n " * [$(basename $doc .txt)](bin/$(basename $doc)): " >>$index
    head -n1 $doc >>$index
  done

  # Bazel build artifacts do not have the write bit set.
  chmod 644 Documentation/bin/*.txt
}
main $@

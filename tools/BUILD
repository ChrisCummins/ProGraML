# This package contains miscellaneous utility scripts.
# Some of these scripts are meant for execution in-tree, and cannot be run
# using bazel, please see the comments.

exports_files([
    "bazel",
])

# Python3 wrapper to workaround docker rules bug.
sh_binary(
    name = "py3_wrapper",
    srcs = ["py3_wrapper.sh"],
    visibility = ["//visibility:public"],
)

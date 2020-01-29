# This package contains miscellaneous utility scripts.
# Some of these scripts are meant for execution in-tree, and cannot be run
# using bazel, please see the comments.

sh_binary(
    name = "whoami",
    srcs = ["whoami.sh"],
    data = [
        "//:build_info_pbtxt",
    ],
)

sh_binary(
    name = "autotex",
    srcs = ["autotex.sh"],
    visibility = ["//visibility:public"],
)

# Python3 wrapper to workaround docker rules bug.
sh_binary(
    name = "py3_wrapper",
    srcs = ["py3_wrapper.sh"],
    visibility = ["//visibility:public"],
)

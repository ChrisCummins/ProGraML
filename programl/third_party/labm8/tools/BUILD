# This package contains miscellaneous utility scripts.
# Some of these scripts are meant for execution in-tree, and cannot be run
# using bazel, please see the comments.

exports_files([
    "bazel",
])

sh_binary(
    name = "whoami",
    srcs = ["whoami.sh"],
    data = [":workspace_status"],
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

genrule(
    name = "workspace_status",
    outs = ["workspace_status.txt"],
    cmd = ("cat bazel-out/volatile-status.txt > $@ &&" +
           "cat bazel-out/stable-status.txt >> $@"),
    stamp = 1,
    visibility = ["//labm8/py:__subpackages__"],
)

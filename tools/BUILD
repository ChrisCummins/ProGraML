# This package contains miscellaneous utility scripts.
# Some of these scripts are meant for execution in-tree, and cannot be run
# using bazel, please see the comments.

sh_binary(
    name = "whoami",
    data = ["//:build_info_pbtxt"],
    srcs = ["whoami.sh"],
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

# Scripts needed by //tools/bzl:maven_jar.bzl. This build target isn't
# required, it is just used here to document the implicit dependency.
filegroup(
    name = "maven_scripts",
    srcs = [
        "download_file.py",
        "util.py",
    ],
)

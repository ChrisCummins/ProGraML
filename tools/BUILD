# This package contains miscellaneous utility scripts.
# Some of these scripts are meant for execution in-tree, and cannot be run
# using bazel, please see the comments.

# Cannot be run using bazel.
sh_binary(
    name = "bootstrap",
    srcs = ["bootstrap.sh"],
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "autotex",
    srcs = ["autotex.sh"],
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "jupyter",
    srcs = ["run_jupyter_notebook.sh"],
    visibility = ["//visibility:public"],
)

# Cannot be run using bazel.
sh_binary(
    name = "buildifier",
    srcs = ["buildifier.sh"],
    visibility = ["//visibility:public"],
)

# Cannot be run using bazel.
sh_binary(
    name = "protoc",
    srcs = ["protoc.sh"],
    visibility = ["//visibility:public"],
)

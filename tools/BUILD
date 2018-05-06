sh_binary(
    name = "autotex",
    srcs = ["autotex.sh"],
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "jupyter",
    srcs = ["run_jupyter_notebook.sh"],
    data = ["//:python_runtime"],
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "buildifier",
    srcs = ["buildifier.sh"],
)

sh_binary(
    name = "protoc",
    srcs = ["protoc.sh"],
)

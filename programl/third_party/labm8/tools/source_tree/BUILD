# Tools for working with the source tree.

load("@subpar//:subpar.bzl", "par_binary")

par_binary(
    name = "deploy_pip",
    srcs = ["deploy_pip.py"],
    data = ["//labm8/py/internal:workspace_status"],
    visibility = ["//visibility:public"],
    deps = [
        ":phd_workspace",
        "//labm8/py:app",
        "//labm8/py:fs",
    ],
)

py_library(
    name = "phd_workspace",
    srcs = ["phd_workspace.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:bazelutil",
        "//labm8/py:fs",
        "//labm8/py:humanize",
        "//third_party/py/git",
        "//tools/git:export_subtree",
    ],
)

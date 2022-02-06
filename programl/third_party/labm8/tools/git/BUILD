# Git utilities.

py_library(
    name = "export_subtree",
    srcs = ["export_subtree.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":git_remote",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:progress",
        "//third_party/py/git",
    ],
)

py_test(
    name = "export_subtree_test",
    srcs = ["export_subtree_test.py"],
    deps = [
        ":export_subtree",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:test",
        "//third_party/py/git",
        "//tools/git/test:fixtures",
    ],
)

py_library(
    name = "git_clone",
    srcs = ["git_clone.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:app",
    ],
)

py_library(
    name = "git_remote",
    srcs = ["git_remote.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:app",
        "//third_party/py/git",
    ],
)

py_test(
    name = "git_remote_test",
    srcs = ["git_remote_test.py"],
    deps = [
        ":git_remote",
        "//labm8/py:test",
        "//third_party/py/git",
        "//tools/git/test:fixtures",
    ],
)

sh_binary(
    name = "repo_stats",
    srcs = ["repo_stats.sh"],
)

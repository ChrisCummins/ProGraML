# Libraries for writing shell scripts and tests.

load("@com_github_chriscummins_rules_bats//:bats.bzl", "bats_test")

sh_library(
    name = "app",
    srcs = ["app.sh"],
    visibility = ["//visibility:public"],
    deps = [
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_library(
    name = "test",
    testonly = 1,
    srcs = ["test.sh"],
    visibility = ["//visibility:public"],
    deps = [":app"],
)

bats_test(
    name = "test_test",
    srcs = ["test_test.bats"],
    data = [":test"],
)

load("@com_github_chriscummins_rules_bats//:bats.bzl", "bats_test")

sh_library(
    name = "test",
    testonly = 1,
    srcs = ["test.sh"],
    visibility = ["//visibility:public"],
)

bats_test(
    name = "test_test",
    srcs = ["test_test.bats"],
    data = [":test"],
)

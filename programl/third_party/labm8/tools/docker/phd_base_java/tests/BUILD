# Tests for phd_base_java docker image.

load("@com_github_chriscummins_rules_bats//:bats.bzl", "bats_test")
load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py_test(
    name = "java_test",
    srcs = ["java_test.py"],
    deps = [
        "//labm8/py:test",
        "//third_party/py/mysql",
        "//third_party/py/numpy",
    ],
)

py3_image(
    name = "java_test_image",
    testonly = 1,
    srcs = ["java_test.py"],
    base = "@phd_base_java//image",
    main = "java_test.py",
    deps = [":java_test"],
)

bats_test(
    name = "tests",
    srcs = ["tests.bats"],
    data = [
        ":java_test_image.tar",
        "//labm8/sh:test",
    ],
)

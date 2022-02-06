# Tests for phd_base_java docker image.

load("@com_github_chriscummins_rules_bats//:bats.bzl", "bats_test")
load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py_test(
    name = "tf_test",
    srcs = ["tf_test.py"],
    deps = [
        "//labm8/py:test",
        "//third_party/py/tensorflow",
    ],
)

py3_image(
    name = "tf_test_image",
    testonly = 1,
    srcs = ["tf_test.py"],
    base = "@phd_base_tf_cpu//image",
    main = "tf_test.py",
    deps = [":tf_test"],
)

bats_test(
    name = "tests",
    srcs = ["tests.bats"],
    data = [
        ":tf_test_image.tar",
        "//labm8/sh:test",
    ],
)

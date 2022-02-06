# Tests for phd_base docker image.

load("@com_github_chriscummins_rules_bats//:bats.bzl", "bats_test")
load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py_test(
    name = "dependencies_test",
    srcs = ["dependencies_test.py"],
    deps = [
        "//labm8/py:test",
        "//third_party/py/mysql",
        "//third_party/py/numpy",
    ],
)

py3_image(
    name = "dependencies_test_image",
    testonly = 1,
    srcs = ["dependencies_test.py"],
    base = "@phd_base//image",
    main = "dependencies_test.py",
    deps = [":dependencies_test"],
)

py_test(
    name = "python_version",
    srcs = ["python_version.py"],
)

py3_image(
    name = "python_version_image",
    testonly = 1,
    srcs = ["python_version.py"],
    base = "@phd_base//image",
    main = "python_version.py",
    deps = [":python_version"],
)

bats_test(
    name = "tests",
    srcs = ["tests.bats"],
    data = [
        ":dependencies_test_image.tar",
        ":python_version_image.tar",
        "//labm8/sh:test",
    ],
)

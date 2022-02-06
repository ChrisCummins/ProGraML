# This package contains a docker image with all project build dependencies.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

exports_files([
    "run.sh",
    "Dockerfile",
    "export.sh",
    "README.md",
])

py3_image(
    name = "phd_build",
    srcs = ["empty.py"],
    base = "@phd_build//image",
    main = "empty.py",
    visibility = ["//visibility:public"],
)

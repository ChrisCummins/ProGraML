# This package defines a Dockerfile image capable of running this project.

# Since the installation is described using a Dockerfile, bazel cannot build
# it.
filegroup(
    name = "phd",
    srcs = ["Dockerfile"],
)

py_binary(
    name = "publish",
    srcs = ["publish.py"],
    data = ["//tools/docker/phd_build:phd_build.tar"],
    deps = [
        "//labm8:app",
        "//labm8:dockerutil",
    ],
)

# This package defines a Dockerfile image capable of running this project.

py_binary(
    name = "publish",
    srcs = ["publish.py"],
    data = ["//tools/docker/phd_build:phd_build.tar"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:dockerutil",
    ],
)

sh_binary(
    name = "remove_all_docker_images",
    srcs = ["remove_all_docker_images.sh"],
)

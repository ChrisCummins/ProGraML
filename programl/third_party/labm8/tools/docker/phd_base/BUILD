# Base image for project docker builds.

exports_files([
    "run.sh",
    "Dockerfile",
    "export.sh",
    "README.md",
])

filegroup(
    name = "phd_base",
    srcs = [
        "Dockerfile",
    ],
    visibility = ["//tools/docker/phd_base:__subpackages__"],
)

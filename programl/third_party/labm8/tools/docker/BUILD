# This package defines a Dockerfile image capable of running this project.

sh_binary(
    name = "export",
    srcs = ["export.sh"],
    data = [
        "//:version.txt",
        "//labm8/sh:app",
    ],
)

sh_binary(
    name = "remove_all_docker_images",
    srcs = ["remove_all_docker_images.sh"],
)

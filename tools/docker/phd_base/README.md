# Docker image: phd_base

A minimal Ubuntu environment with runtime dependencies for bazel-compiled
binaries.


## Usage

Start an interactive bash session using: `./tools/docker/phd_base/run.sh`.

To make an image from a python target, add to your `BUILD` file:

```sh
load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py3_image(
    name = "image",
    srcs = [":foo"],
    main = ["foo.py"],
    base = "@phd_base//image",
    deps = [":foo"],
)
```


## Updating [chriscummins/phd_base](https://hub.docker.com/r/chriscummins/phd_base)

1. Modify `tools/docker/phd_base/Dockerfile` with your desired changes.
2. Build and export the image using: `$ ./tools/docker/phd_base/export.sh`.
3. Update the digest in the `WORKSPACE` file:
```
container_pull(
    name = "phd_base",
    digest = "sha256:<new_digest>",
    registry = "index.docker.io",
    repository = "chriscummins/phd_base",
)
```
4. Update the version tag in `phd_java` docker image.

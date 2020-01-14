# Docker image: phd_build

A self-contained environment configured to build and run the code in this
project.


## Usage

Start an interactive bash session using: `./tools/docker/phd_build/run.sh`.

To make an image from a python target, add to your `BUILD` file:

```sh
load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py3_image(
    name = "image",
    srcs = [":foo"],
    main = ["foo.py"],
    base = "@phd_build//image",
    deps = [":foo"],
)
```



## Updating [chriscummins/phd_build](https://hub.docker.com/r/chriscummins/phd_build)

1. Set a new project version in `//:version.txt`.
2. Modify `tools/docker/phd_build/Dockerfile` with your desired changes.
3. Build and export the image using: `$ bazel run //tools/docker:export -- $PHD/tools/docker/phd_build`.
4. Update the digest in the `WORKSPACE` file:
```
container_pull(
    name = "phd_build",
    digest = "sha256:<new_digest>",
    registry = "index.docker.io",
    repository = "chriscummins/phd_build",
)
```

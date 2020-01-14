# Docker image: phd_base_java

An extension to `phd_base` with Java.

To create a docker image from a `py_binary` target, create a `py3_image` target
and set the `base` to `@base//image`, e.g:

```sh
load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py3_image(
    name = "image",
    srcs = [":foo"],
    main = ["foo.py"],
    base = "@phd_base_java//image",
    deps = [":foo"],
)
```

## Updating [chriscummins/phd_base_java](https://hub.docker.com/r/chriscummins/phd_base_java)

1. Set a new project version in `//:version.txt`.
2. Modify `tools/docker/phd_base/Dockerfile` with your desired changes.
3. Build and export the image using: `$ bazel run //tools/docker:export -- $PHD/tools/docker/phd_base_java`.
4. Update the digest in the `WORKSPACE` file:
```
container_pull(
    name = "phd_base_java",
    digest = "sha256:<new_digest>",
    registry = "index.docker.io",
    repository = "chriscummins/phd_base_java",
)
```
5. Run the test suite: `$ bazel test //tools/docker/phd_base_java/...`.
6. Update the base tags in `//tools/docker/phd_build:Dockerfile`.

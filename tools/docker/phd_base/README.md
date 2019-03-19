# Docker image: phd_base

A minimal Ubuntu environment with runtime dependencies for bazel-compiled
binaries.

Build using:

```sh
$ docker build -t phd_base --squash $PHD/tools/docker/phd_base
```

To create a docker image from a `py_binary` target, create a `py3_image` target 
and set the `base` to `@base//image`, e.g:  

```sh
load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py3_image(
    name = "image",
    srcs = [":foo"],
    main = ["foo.py"],
    base = "@base//image",
    deps = [":foo"],
)
```

To update the `chriscummins/phd_base` image used in this project:

1. Create a named tag for the build:
   `docker tag phd_base chriscummins/phd_base:VERSION`.
1. Push the tag to docker hub: `docker push chriscummins/phd_base:VERSION`.
1. Update the `digest` attribute of the `container_pull` command that defines
   the base container in `//:WORKSPACE`.

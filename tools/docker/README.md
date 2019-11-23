# Docker images

This package contains docker images for to be by bazel for creating dockerized 
builds, for example:

```
load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py3_image(
    name = "foo_image",
    srcs = ["foo.py"],
    base = "@base//image",
    main = "foo.py",
    deps = [":foo"],
)
```

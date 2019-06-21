load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py3_image(
    name = "phd_build",
    srcs = ["empty.py"],
    base = "@phd_build//image",
    main = "empty.py",
    visibility = ["//visibility:public"],
)

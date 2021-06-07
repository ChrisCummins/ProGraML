py_library(
    name = "programl",
    srcs = [
        "__init__.py",
        "create_ops.py",
        "serialization_ops.py",
        "transform_ops.py",
        "version.py",
    ],
    data = [
        "//programl/bin",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/proto",
        "//programl/third_party/tensorflow:xla_py",
        "//programl/util/py:util",
    ],
)

genrule(
    name = "version_header-gen",
    srcs = ["//:version.txt"],
    outs = ["version.h"],
    cmd = "echo \"#define PROGRAML_VERSION \\\"$$(tr -d '\n' < $<)\\\"\" > $@",
)

genrule(
    name = "version_py-gen",
    srcs = ["//:version.txt"],
    outs = ["version.py"],
    cmd = "echo \"PROGRAML_VERSION=\\\"$$(tr -d '\n' < $<)\\\"\" > $@",
)

cc_library(
    name = "version_header",
    hdrs = ["version.h"],
    visibility = ["//visibility:public"],
)

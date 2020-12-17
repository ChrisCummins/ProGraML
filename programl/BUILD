genrule(
    name = "version_header-gen",
    srcs = ["//:version.txt"],
    outs = ["version.h"],
    cmd = "echo \"#define PROGRAML_VERSION \\\"$$(tr -d '\n' < $<)\\\"\" > $@",
)

genrule(
    name = "version_py-gen",
    srcs = ["//:version.txt"],
    outs = ["__init__.py"],
    cmd = "echo \"PROGRAML_VERSION=\\\"$$(tr -d '\n' < $<)\\\"\" > $@",
)

cc_library(
    name = "version_header",
    hdrs = ["version.h"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "programl",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
)

# Top level package of the phd repo.

load("//tools/bzl:string_genrule.bzl", "python_string_genrule")
load("@bazel_gazelle//:def.bzl", "gazelle")
load("@build_stack_rules_proto//python:python_grpc_library.bzl", "python_grpc_library")

exports_files([
    "README.md",
    "version.txt",
    "deployment.properties",
])

python_string_genrule(
    name = "version_py",
    src = "version.txt",
)

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

py_library(
    name = "build_info",
    srcs = ["build_info.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//:build_info_pbtxt_py",
        "//:config_pb_py",
        "//:version_py",
        "//labm8/py:pbutil",
    ],
)

py_test(
    name = "build_info_test",
    srcs = ["build_info_test.py"],
    deps = [
        ":build_info",
        "//labm8/py:test",
    ],
)

genrule(
    name = "build_info_pbtxt",
    outs = ["build_info.pbtxt"],
    cmd = "$(location :make_build_info_pbtxt) > $@",
    stamp = 1,
    tools = [":make_build_info_pbtxt"],
    visibility = ["//visibility:public"],
)

python_string_genrule(
    name = "build_info_pbtxt_py",
    src = ":build_info.pbtxt",
)

sh_binary(
    name = "make_build_info_pbtxt",
    srcs = ["make_build_info_pbtxt.sh"],
)

proto_library(
    name = "config_pb",
    srcs = ["config.proto"],
    visibility = ["//visibility:public"],
)

# Use GRPC so that dependent targets can be GRPC libraries. Otherwise, if
# a python_proto_library, there is a conflict.
python_grpc_library(
    name = "config_pb_py",
    visibility = ["//visibility:public"],
    deps = ["//:config_pb"],
)

# Golang.
# Gazelle directive:
# gazelle:prefix github.com/ChrisCummins/phd
# gazelle:proto disable

gazelle(name = "gazelle")

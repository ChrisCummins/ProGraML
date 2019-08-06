# Top level package of the phd repo.

load("//tools/bzl:string_genrule.bzl", "python_string_genrule")
load("@bazel_gazelle//:def.bzl", "gazelle")
load("@build_stack_rules_proto//python:python_grpc_library.bzl", "python_grpc_library")

exports_files([
    "README.md",
    "version.txt",
    "deployment.properties",
])

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
        "//labm8:pbutil",
    ],
)

py_test(
    name = "build_info_test",
    srcs = ["build_info_test.py"],
    deps = [
        ":build_info",
        "//labm8:test",
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
    out = "build_info_pbtxt.py",
)

sh_binary(
    name = "make_build_info_pbtxt",
    srcs = ["make_build_info_pbtxt.sh"],
)

filegroup(
    name = "config",
    srcs = ["config.pbtxt"],
    visibility = ["//visibility:public"],
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

python_string_genrule(
    name = "config_pbtxt_py",
    src = ":config",
    out = "config_pbtxt.py",
)

filegroup(
    name = "configure_py",
    srcs = ["configure"],
)

py_library(
    name = "conftest",
    testonly = 1,
    srcs = ["conftest.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//:build_info",
        "//labm8:app",
        "//third_party/py/pytest",
    ],
)

py_test(
    name = "configure_test",
    srcs = ["configure_test.py"],
    data = [":configure_py"],
    deps = [
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:test",
    ],
)

py_library(
    name = "getconfig",
    srcs = ["getconfig.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//:config_pb_py",
        "//:config_pbtxt_py",
        "//labm8:pbutil",
    ],
)

py_test(
    name = "getconfig_test",
    srcs = ["getconfig_test.py"],
    deps = [
        ":getconfig",
        "//labm8:app",
        "//labm8:test",
    ],
)

# Golang.
# Gazelle directive:
# gazelle:prefix github.com/ChrisCummins/phd
# gazelle:proto disable

gazelle(name = "gazelle")

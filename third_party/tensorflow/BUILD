load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

proto_library(
    name = "xla_data",
    srcs = ["compiler/xla/xla_data.proto"],
)

proto_library(
    name = "xla_protos",
    srcs = ["compiler/xla/service/hlo.proto"],
    visibility = ["//visibility:public"],
    deps = [":xla_data"],
)

py_proto_library(
    name = "xla_protos_py",
    visibility = ["//visibility:public"],
    deps = [":xla_protos"],
)

cc_proto_library(
    name = "xla_protos_cc",
    visibility = ["//visibility:public"],
    deps = [":xla_protos"],
)

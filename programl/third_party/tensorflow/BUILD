load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

proto_library(
    name = "features",
    srcs = ["features.proto"],
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "features_cc",
    visibility = ["//visibility:public"],
    deps = [":features"],
)

py_proto_library(
    name = "features_py",
    visibility = ["//visibility:public"],
    deps = [":features"],
)

proto_library(
    name = "xla",
    srcs = ["xla.proto"],
    visibility = ["//visibility:public"],
)

py_proto_library(
    name = "xla_py",
    visibility = ["//visibility:public"],
    deps = [":xla"],
)

cc_proto_library(
    name = "xla_cc",
    visibility = ["//visibility:public"],
    deps = [":xla"],
)

# Test data for //labm8/py:ppar_test.

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

package(default_visibility = ["//labm8/py:__subpackages__"])

proto_library(
    name = "protos",
    srcs = ["protos.proto"],
)

cc_proto_library(
    name = "protos_pb_cc",
    deps = [":protos"],
)

python_proto_library(
    name = "protos_pb_py",
    deps = [":protos"],
)

cc_binary(
    name = "proto_worker",
    srcs = ["proto_worker.cc"],
    deps = [
        ":protos_pb_cc",
        "//labm8/cpp:logging",
        "//labm8/cpp:pbutil",
    ],
)

cc_binary(
    name = "proto_worker_requires_args",
    srcs = ["proto_worker_requires_args.cc"],
    deps = [
        "//labm8/cpp:logging",
    ],
)

cc_binary(
    name = "unexpected_output_proto_worker",
    srcs = ["unexpected_output_proto_worker.cc"],
    deps = [
        ":protos_pb_cc",
        "//labm8/cpp:logging",
    ],
)

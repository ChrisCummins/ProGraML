# Protocol message schemas for ProGraML.
#
# Copyright 2019-2020 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_proto_library")

proto_library(
    name = "programl",
    srcs = [
        "checkpoint.proto",
        "edge.proto",
        "epoch.proto",
        "features.proto",
        "function.proto",
        "ir.proto",
        "module.proto",
        "node.proto",
        "program_graph.proto",
        "program_graph_features.proto",
        "program_graph_options.proto",
        "repo.proto",
        "src.proto",
    ],
    visibility = ["//visibility:public"],
)

py_proto_library(
    name = "programl_py",
    visibility = ["//visibility:public"],
    deps = [":programl"],
)

cc_proto_library(
    name = "programl_cc",
    visibility = ["//visibility:public"],
    deps = [":programl"],
)

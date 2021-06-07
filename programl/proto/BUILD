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

py_library(
    name = "proto",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":programl_py",
        "//programl/third_party/tensorflow:features_py",
    ],
)

proto_library(
    name = "programl",
    srcs = [
        "program_graph.proto",
        "util.proto",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/third_party/tensorflow:features",
    ],
)

py_proto_library(
    name = "programl_py",
    deps = [":programl"],
)

cc_proto_library(
    name = "programl_cc",
    visibility = ["//visibility:public"],
    deps = [":programl"],
)

# This package contains python bindings for graph format converters.
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

load(
    "//third_party/pybind11_bazel:build_defs.bzl",
    "pybind_extension",
)

py_library(
    name = "cdfg",
    srcs = ["cdfg.py"],
    data = [
        "//programl/cmd:graph2cdfg",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:bazelutil",
        "//programl/proto:program_graph_py",
    ],
)

py_library(
    name = "graph_serializer",
    srcs = ["graph_serializer.py"],
    data = [":graph_serializer_pybind.so"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/proto:program_graph_py",
    ],
)

py_test(
    name = "graph_serializer_test",
    srcs = ["graph_serializer_test.py"],
    deps = [
        ":graph_serializer",
        "//labm8/py:test",
        "//programl/proto:edge_py",
        "//programl/proto:node_py",
        "//programl/proto:program_graph_py",
    ],
)

pybind_extension(
    "graph_serializer_pybind",
    srcs = ["graph_serializer_pybind.cc"],
    deps = [
        "//labm8/cpp:status",
        "//labm8/cpp:string",
        "//programl/graph/format:graph_serializer",
        "//programl/proto:program_graph_cc",
    ],
)

py_library(
    name = "graph_tuple",
    srcs = ["graph_tuple.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "graph_tuple_builder",
    srcs = ["graph_tuple_builder.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_tuple",
        "//programl/proto:program_graph_py",
        "//third_party/py/numpy",
    ],
)

pybind_extension(
    "graph_tuple_pybind",
    srcs = ["graph_tuple_pybind.cc"],
    deps = [
        "//programl/graph/format:graph_tuple",
    ],
)

py_test(
    name = "graph_tuple_builder_test",
    srcs = ["graph_tuple_builder_test.py"],
    deps = [
        ":graph_tuple_builder",
        "//labm8/py:test",
        "//programl/proto:program_graph_py",
    ],
)

pybind_extension(
    "node_link_graph_pybind",
    srcs = ["node_link_graph_pybind.cc"],
    deps = [
        "//labm8/cpp:status",
        "//labm8/cpp:string",
        "//programl/graph/format:node_link_graph",
        "//programl/proto:program_graph_cc",
        "@nlohmann_json//:json",
        "@pybind11_json",
    ],
)

py_library(
    name = "nx_format",
    srcs = ["nx_format.py"],
    data = [":node_link_graph_pybind.so"],
    visibility = ["//programl:__subpackages__"],
    deps = [
        "//programl/proto:program_graph_py",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "nx_format_test",
    srcs = ["nx_format_test.py"],
    deps = [
        ":nx_format",
        "//labm8/py:test",
        "//programl/proto:program_graph_py",
        "//third_party/py/networkx",
    ],
)

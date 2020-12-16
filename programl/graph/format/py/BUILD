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
        "//bin:graph2cdfg",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/proto:programl_py",
        "//programl/util/py:runfiles_path",
    ],
)

py_library(
    name = "graph_serializer",
    srcs = ["graph_serializer.py"],
    data = [":graph_serializer_pybind.so"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/proto:programl_py",
    ],
)

pybind_extension(
    "graph_serializer_pybind",
    srcs = ["graph_serializer_pybind.cc"],
    deps = [
        "//programl/graph/format:graph_serializer",
        "//programl/proto:programl_cc",
        "@labm8//labm8/cpp:status",
        "@labm8//labm8/cpp:string",
    ],
)

py_library(
    name = "graph_tuple",
    srcs = ["graph_tuple.py"],
    visibility = ["//visibility:public"],
    deps = [
    ],
)

py_library(
    name = "graph_tuple_builder",
    srcs = ["graph_tuple_builder.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_tuple",
        "//programl/proto:programl_py",
    ],
)

pybind_extension(
    "graph_tuple_pybind",
    srcs = ["graph_tuple_pybind.cc"],
    deps = [
        "//programl/graph/format:graph_tuple",
    ],
)

pybind_extension(
    "node_link_graph_pybind",
    srcs = ["node_link_graph_pybind.cc"],
    deps = [
        "//programl/graph/format:node_link_graph",
        "//programl/proto:programl_cc",
        "@labm8//labm8/cpp:status",
        "@labm8//labm8/cpp:string",
        "@nlohmann_json//:json",
        "@pybind11_json",
    ],
)

py_library(
    name = "nx_format",
    srcs = ["nx_format.py"],
    data = [":node_link_graph_pybind.so"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/proto:programl_py",
    ],
)

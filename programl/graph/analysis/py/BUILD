# Python bindings for analyses.
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
    name = "analysis",
    srcs = ["analysis.py"],
    data = [":analysis_pybind.so"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/proto:program_graph_features_py",
        "//programl/proto:program_graph_py",
    ],
)

pybind_extension(
    name = "analysis_pybind",
    srcs = ["analysis_pybind.cc"],
    deps = [
        "//labm8/cpp:status",
        "//labm8/cpp:string",
        "//programl/graph:program_graph_builder",
        "//programl/graph/analysis",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_features_cc",
    ],
)

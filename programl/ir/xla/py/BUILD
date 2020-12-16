# This package contains python bindings for xla.
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
    name = "xla",
    srcs = ["xla.py"],
    data = [":xla_pybind.so"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/proto:programl_py",
        "//third_party/tensorflow:xla_py",
    ],
)

pybind_extension(
    "xla_pybind",
    srcs = ["xla_pybind.cc"],
    deps = [
        "//programl/ir/xla:hlo_module_graph_builder",
        "//third_party/tensorflow:xla_cc",
        "@labm8//labm8/cpp:statusor",
        "@labm8//labm8/cpp:string",
    ],
)

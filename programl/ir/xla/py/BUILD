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
    deps = [
        "//labm8/py:app",
        "//programl/proto:program_graph_py",
        "//third_party/py/networkx",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto_lib_py",
    ],
)

py_test(
    name = "xla_test",
    srcs = ["xla_test.py"],
    data = ["//programl/test/data:hlo_protos"],
    deps = [
        ":xla",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:pbutil",
        "//labm8/py:test",
        "@org_tensorflow//tensorflow/compiler/xla:xla_data_proto_lib_py",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto_lib_py",
    ],
)

pybind_extension(
    "xla_pybind",
    srcs = ["xla_pybind.cc"],
    deps = [
        "//labm8/cpp:statusor",
        "//labm8/cpp:string",
        "//programl/ir/xla:hlo_module_graph_builder",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto",
    ],
)

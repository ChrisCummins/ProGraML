# This package generates program graphs from XLA HLO modules.
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

cc_library(
    name = "hlo_module_graph_builder",
    srcs = ["hlo_module_graph_builder.cc"],
    hdrs = ["hlo_module_graph_builder.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":xla_stringifier",
        "//labm8/cpp:logging",
        "//labm8/cpp:port",
        "//labm8/cpp:status_macros",
        "//labm8/cpp:statusor",
        "//labm8/cpp:string",
        "//programl/graph:program_graph_builder",
        "//programl/proto:program_graph_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@org_tensorflow//tensorflow/compiler/xla:xla_data_proto",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto",
    ],
)

cc_library(
    name = "hlo_proto_reader",
    srcs = ["hlo_proto_reader.cc"],
    hdrs = ["hlo_proto_reader.h"],
    deps = [
        "//labm8/cpp:status_macros",
        "//labm8/cpp:statusor",
        "//labm8/cpp:string",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto",
    ],
)

cc_test(
    name = "hlo_proto_reader_test",
    srcs = ["hlo_proto_reader_test.cc"],
    deps = [
        ":hlo_proto_reader",
        "//labm8/cpp:test",
        "@boost//:filesystem",
    ],
)

cc_library(
    name = "xla_stringifier",
    srcs = ["xla_stringifier.cc"],
    hdrs = ["xla_stringifier.h"],
    deps = [
        "//labm8/cpp:string",
        "@org_tensorflow//tensorflow/compiler/xla:xla_data_proto",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto",
    ],
)

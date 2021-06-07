# This package contains code for working with the ProGraML representation.
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
    name = "features",
    srcs = ["features.cc"],
    hdrs = ["features.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/third_party/tensorflow:features_cc",
    ],
)

cc_library(
    name = "program_graph_builder",
    srcs = ["program_graph_builder.cc"],
    hdrs = ["program_graph_builder.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/proto:programl_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@labm8//labm8/cpp:logging",
        "@labm8//labm8/cpp:status",
        "@labm8//labm8/cpp:status_macros",
        "@labm8//labm8/cpp:statusor",
        "@labm8//labm8/cpp:string",
    ],
)

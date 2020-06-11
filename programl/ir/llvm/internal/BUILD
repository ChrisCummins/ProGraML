# This package contains implementation details for LLVM-IR support.
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

filegroup(
    name = "inst2vec_augmented_dictionary",
    srcs = ["inst2vec_augmented_dictionary.pickle"],
    visibility = ["//programl/ir/llvm:__subpackages__"],
)

filegroup(
    name = "inst2vec_augmented_embeddings",
    srcs = ["inst2vec_augmented_embeddings.pickle"],
    visibility = ["//programl/ir/llvm:__subpackages__"],
)

cc_library(
    name = "program_graph_builder",
    srcs = ["program_graph_builder.cc"],
    hdrs = ["program_graph_builder.h"],
    copts = ["-std=c++14"],
    deps = [
        ":text_encoder",
        "//labm8/cpp:status_macros",
        "//labm8/cpp:statusor",
        "//labm8/cpp:string",
        "//programl/graph:features",
        "//programl/graph:program_graph_builder",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_options_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm//10.0.0",
    ],
)

cc_library(
    name = "program_graph_builder_pass",
    srcs = ["program_graph_builder_pass.cc"],
    hdrs = ["program_graph_builder_pass.h"],
    copts = [
        "-std=c++14",
        "-fno-rtti",
        "-DGOOGLE_PROTOBUF_NO_RTTI",
    ],
    visibility = ["//programl/ir/llvm:__subpackages__"],
    deps = [
        ":program_graph_builder",
        "//labm8/cpp:statusor",
        "//programl/proto:program_graph_options_cc",
        "@llvm//10.0.0",
    ],
)

cc_library(
    name = "text_encoder",
    srcs = ["text_encoder.cc"],
    hdrs = ["text_encoder.h"],
    copts = ["-std=c++14"],
    deps = [
        "//labm8/cpp:logging",
        "//labm8/cpp:string",
        "@com_google_absl//absl/container:flat_hash_map",
        "@llvm//10.0.0",
    ],
)

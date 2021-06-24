# This package implements support for LLVM-IR.
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
    name = "clang",
    srcs = ["clang.cc"],
    hdrs = ["clang.h"],
    data = select({
        "//:darwin": [
            "@clang-llvm-10.0.0-x86_64-apple-darwin//:clang++",
        ],
        "//conditions:default": [
            "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang++",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//programl/proto:programl_cc",
        "@com_google_absl//absl/strings:str_format",
        "@labm8//labm8/cpp:bazelutil",
        "@labm8//labm8/cpp:status",
        "@labm8//labm8/cpp:string",
        "@subprocess",
    ],
)

filegroup(
    name = "inst2vec_augmented_dictionary",
    srcs = ["internal/inst2vec_augmented_dictionary.pickle"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "inst2vec_augmented_embeddings",
    srcs = ["internal/inst2vec_augmented_embeddings.pickle"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "inst2vec_encoder",
    srcs = ["inst2vec_encoder.py"],
    data = [
        ":inst2vec_augmented_dictionary",
        ":inst2vec_augmented_embeddings",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/proto",
        "//programl/third_party/inst2vec",
        "//programl/util/py:decorators",
        "//programl/util/py:pbutil",
        "//programl/util/py:progress",
        "//programl/util/py:runfiles_path",
    ],
)

# The following cc_library definitions are identical except for the @llvm
# dependency version.
#
# TODO(cummins): Find a neater way to declare these.

cc_library(
    name = "llvm-3.8",
    srcs = ["llvm.cc"] + glob(["internal/*.cc"]),
    hdrs = ["llvm.h"] + glob(["internal/*.h"]),
    copts = [
        "-std=c++14",
        "-fno-rtti",
    ],
    defines = [
        "GOOGLE_PROTOBUF_NO_RTTI",
        "PROGRAML_LLVM_VERSION_MAJOR=3",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/graph:features",
        "//programl/graph:program_graph_builder",
        "//programl/proto:programl_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@labm8//labm8/cpp:status",
        "@labm8//labm8/cpp:status_macros",
        "@labm8//labm8/cpp:statusor",
        "@labm8//labm8/cpp:string",
        "@llvm//3.8.0",
    ],
)

cc_library(
    name = "llvm-6",
    srcs = ["llvm.cc"] + glob(["internal/*.cc"]),
    hdrs = ["llvm.h"] + glob(["internal/*.h"]),
    copts = [
        "-std=c++14",
        "-fno-rtti",
    ],
    defines = [
        "GOOGLE_PROTOBUF_NO_RTTI",
        "PROGRAML_LLVM_VERSION_MAJOR=6",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/graph:features",
        "//programl/graph:program_graph_builder",
        "//programl/proto:programl_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@labm8//labm8/cpp:status",
        "@labm8//labm8/cpp:status_macros",
        "@labm8//labm8/cpp:statusor",
        "@labm8//labm8/cpp:string",
        "@llvm//6.0.0",
    ],
)

cc_library(
    name = "llvm-10",
    srcs = ["llvm.cc"] + glob(["internal/*.cc"]),
    hdrs = ["llvm.h"] + glob(["internal/*.h"]),
    copts = [
        "-std=c++14",
        "-fno-rtti",
    ],
    defines = [
        "GOOGLE_PROTOBUF_NO_RTTI",
        "PROGRAML_LLVM_VERSION_MAJOR=10",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/graph:features",
        "//programl/graph:program_graph_builder",
        "//programl/proto:programl_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@labm8//labm8/cpp:status",
        "@labm8//labm8/cpp:status_macros",
        "@labm8//labm8/cpp:statusor",
        "@labm8//labm8/cpp:string",
        "@llvm//10.0.0",
    ],
)

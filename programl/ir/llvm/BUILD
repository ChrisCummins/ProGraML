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

py_library(
    name = "inst2vec_encoder",
    srcs = ["inst2vec_encoder.py"],
    data = [
        "//programl/ir/llvm/internal:inst2vec_augmented_dictionary",
        "//programl/ir/llvm/internal:inst2vec_augmented_embeddings",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/graph/py:program_graph_builder",
        "//programl/proto:programl_py",
        "//programl/util/py:decorators",
        "//programl/util/py:pbutil",
        "//programl/util/py:progress",
        "//programl/util/py:runfiles_path",
        "//third_party/ncc/inst2vec:inst2vec_preprocess",
    ],
)

cc_library(
    name = "llvm",
    srcs = ["llvm.cc"],
    hdrs = ["llvm.h"],
    copts = ["-std=c++14"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/ir/llvm/internal:program_graph_builder_pass",
        "//programl/proto:programl_cc",
        "@labm8//labm8/cpp:status",
        "@labm8//labm8/cpp:status_macros",
        "@llvm//10.0.0",
    ],
)

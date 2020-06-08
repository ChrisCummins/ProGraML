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
            "@llvm_mac//:clang++",
        ],
        "//conditions:default": [
            "@llvm_linux//:clang++",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/cpp:bazelutil",
        "//labm8/cpp:status",
        "//labm8/cpp:string",
        "//programl/proto:ir_cc",
        "@com_google_absl//absl/strings:str_format",
        "@subprocess",
    ],
)

cc_test(
    name = "clang_test",
    srcs = ["clang_test.cc"],
    deps = [
        "//labm8/cpp:test",
        "//programl/ir/llvm:clang",
        "//programl/proto:ir_cc",
    ],
)

py_library(
    name = "inst2vec_encoder",
    srcs = ["inst2vec_encoder.py"],
    data = [
        "//programl/ir/llvm/internal:inst2vec_augmented_dictionary",
        "//programl/ir/llvm/internal:inst2vec_augmented_embeddings",
    ],
    visibility = ["//programl:__subpackages__"],
    deps = [
        "//deeplearning/ncc/inst2vec:inst2vec_preprocess",
        "//labm8/py:bazelutil",
        "//labm8/py:decorators",
        "//labm8/py:labtypes",
        "//labm8/py:pbutil",
        "//labm8/py:progress",
        "//programl/graph/py:program_graph_builder",
        "//programl/proto:node_py",
        "//programl/proto:program_graph_py",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "inst2vec_encoder_test",
    srcs = ["inst2vec_encoder_test.py"],
    deps = [
        "//labm8/py:test",
        "//programl/graph/py:program_graph_builder",
        "//programl/ir/llvm:inst2vec_encoder",
        "//programl/proto:node_py",
        "//programl/proto:program_graph_py",
        "//programl/test/py/plugins:llvm_program_graph",
    ],
)

cc_library(
    name = "llvm",
    srcs = ["llvm.cc"],
    hdrs = ["llvm.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/cpp:status",
        "//labm8/cpp:status_macros",
        "//programl/ir/llvm/internal:program_graph_builder_pass",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_options_cc",
        "//third_party/llvm",
    ],
)

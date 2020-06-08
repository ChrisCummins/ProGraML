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

py_binary(
    name = "collect_stats",
    srcs = ["collect_stats.py"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:pbutil",
        "//labm8/py:progress",
        "//programl/proto:program_graph_features_py",
        "//programl/proto:program_graph_py",
        "//programl/task/dataflow/dataset:pathflag",
    ],
)

py_binary(
    name = "create",
    srcs = ["create.py"],
    data = [
        ":create_labels",
        ":create_vocab",
        ":unpack_ir_lists",
    ],
    deps = [
        ":encode_inst2vec",
        ":pathflag",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:labtypes",
        "//labm8/py:pbutil",
        "//labm8/py:progress",
        "//programl/ir/llvm/py:llvm",
        "//programl/proto:ir_py",
        "//third_party/py/mysql",
    ],
)

sh_binary(
    name = "count_instances",
    srcs = ["count_instances.sh"],
    data = [
        "//labm8/sh:app",
        "//programl/cmd:pbq",
    ],
)

sh_binary(
    name = "count_labels",
    srcs = ["count_labels.sh"],
    data = [
        "//labm8/sh:app",
        "//programl/cmd:pbq",
    ],
)

sh_binary(
    name = "create_labels",
    srcs = ["create_labels.sh"],
    data = [
        "//labm8/sh:app",
        "//programl/cmd:analyze",
    ],
)

cc_binary(
    name = "create_vocab",
    srcs = ["create_vocab.cc"],
    deps = [
        "//labm8/cpp:app",
        "//labm8/cpp:fsutil",
        "//labm8/cpp:logging",
        "//labm8/cpp:strutil",
        "//programl/proto:program_graph_cc",
        "@boost//:filesystem",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
    ],
)

py_binary(
    name = "encode_inst2vec",
    srcs = ["encode_inst2vec.py"],
    deps = [
        ":pathflag",
        "//labm8/py:app",
        "//labm8/py:decorators",
        "//labm8/py:labtypes",
        "//labm8/py:pbutil",
        "//labm8/py:progress",
        "//programl/ir/llvm:inst2vec_encoder",
        "//programl/proto:ir_py",
        "//programl/proto:program_graph_py",
    ],
)

cc_library(
    name = "parallel_file_map",
    srcs = ["parallel_file_map.cc"],
    hdrs = ["parallel_file_map.h"],
    copts = ["-fopenmp"],
    deps = [
        "//labm8/cpp:app",
        "//labm8/cpp:logging",
        "//labm8/cpp:strutil",
        "//programl/graph/analysis:dominance",
        "//programl/graph/analysis:reachability",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_features_cc",
        "//third_party/openmp",
        "//third_party/tbb",
        "@boost//:filesystem",
        "@com_google_absl//absl/strings:str_format",
    ],
)

py_library(
    name = "pathflag",
    srcs = ["pathflag.py"],
    visibility = ["//programl/task/dataflow:__subpackages__"],
    deps = [
        "//labm8/py:app",
    ],
)

cc_binary(
    name = "unpack_ir_lists",
    srcs = ["unpack_ir_lists.cc"],
    copts = ["-fopenmp"],
    deps = [
        ":parallel_file_map",
        "//labm8/cpp:app",
        "//labm8/cpp:fsutil",
        "//labm8/cpp:logging",
        "//labm8/cpp:strutil",
        "//programl/graph/format:cdfg",
        "//programl/proto:ir_cc",
        "//third_party/openmp",
        "@boost//:filesystem",
        "@com_google_absl//absl/strings:str_format",
    ],
)

sh_binary(
    name = "unpack_ir_text",
    srcs = ["unpack_ir_text.sh"],
    data = [
        "//labm8/sh:app",
        "//programl/cmd:pbq",
    ],
)

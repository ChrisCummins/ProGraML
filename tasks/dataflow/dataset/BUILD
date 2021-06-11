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
        "//programl/proto",
        "//programl/util/py:pbutil",
        "//programl/util/py:progress",
        "//tasks/dataflow/dataset:pathflag",
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
        "//programl",
    ],
)

sh_binary(
    name = "count_instances",
    srcs = ["count_instances.sh"],
    data = [
        "//programl/bin:pbq",
        "@labm8//labm8/sh:app",
    ],
)

sh_binary(
    name = "count_labels",
    srcs = ["count_labels.sh"],
    data = [
        "//programl/bin:pbq",
        "@labm8//labm8/sh:app",
    ],
)

sh_binary(
    name = "create_labels",
    srcs = ["create_labels.sh"],
    data = [
        "//programl/bin:analyze",
        "@labm8//labm8/sh:app",
    ],
)

cc_binary(
    name = "create_vocab",
    srcs = ["create_vocab.cc"],
    deps = [
        "//programl/proto:programl_cc",
        "@boost//:filesystem",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@labm8//labm8/cpp:app",
        "@labm8//labm8/cpp:fsutil",
        "@labm8//labm8/cpp:logging",
        "@labm8//labm8/cpp:strutil",
    ],
)

py_binary(
    name = "encode_inst2vec",
    srcs = ["encode_inst2vec.py"],
    deps = [
        ":pathflag",
        "//programl/ir/llvm:inst2vec_encoder",
        "//programl/proto",
        "//programl/util/py:decorators",
        "//programl/util/py:pbutil",
        "//programl/util/py:progress",
    ],
)

cc_library(
    name = "parallel_file_map",
    srcs = ["parallel_file_map.cc"],
    hdrs = ["parallel_file_map.h"],
    deps = [
        "//programl/graph/analysis:dominance",
        "//programl/graph/analysis:reachability",
        "//programl/proto:programl_cc",
        "//third_party/tbb",
        "@boost//:filesystem",
        "@com_google_absl//absl/strings:str_format",
        "@labm8//labm8/cpp:app",
        "@labm8//labm8/cpp:logging",
        "@labm8//labm8/cpp:strutil",
    ],
)

py_library(
    name = "pathflag",
    srcs = ["pathflag.py"],
    visibility = ["//tasks:__subpackages__"],
    deps = [
    ],
)

cc_binary(
    name = "unpack_ir_lists",
    srcs = ["unpack_ir_lists.cc"],
    deps = [
        ":parallel_file_map",
        "//programl/graph/format:cdfg",
        "//programl/proto:programl_cc",
        "@boost//:filesystem",
        "@com_google_absl//absl/strings:str_format",
        "@labm8//labm8/cpp:app",
        "@labm8//labm8/cpp:fsutil",
        "@labm8//labm8/cpp:logging",
        "@labm8//labm8/cpp:strutil",
    ],
)

sh_binary(
    name = "unpack_ir_text",
    srcs = ["unpack_ir_text.sh"],
    data = [
        "//programl/bin:pbq",
        "@labm8//labm8/sh:app",
    ],
)

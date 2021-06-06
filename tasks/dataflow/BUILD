# This package contains the code for learning data flow analyses.
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

py_library(
    name = "ggnn",
    srcs = ["ggnn.py"],
    deps = [
        ":dataflow",
        ":ggnn_batch_builder",
        ":graph_loader",
        "//models:async_batch_builder",
        "//models:epoch_batch_iterator",
        "//models/ggnn",
        "//programl/proto:programl_py",
        "//programl/util/py:humanize",
        "//programl/util/py:pbutil",
    ],
)

py_library(
    name = "ggnn_batch_builder",
    srcs = ["ggnn_batch_builder.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//models:base_batch_builder",
        "//models:base_graph_loader",
        "//models:batch_data",
        "//models/ggnn:ggnn_batch",
        "//programl/graph/format/py:graph_tuple_builder",
    ],
)

py_library(
    name = "dataflow",
    srcs = ["dataflow.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_loader",
        "//models:async_batch_builder",
        "//models/ggnn",
        "//programl/proto:programl_py",
        "//programl/util/py:pbutil",
    ],
)

py_binary(
    name = "export_logs",
    srcs = ["export_logs.py"],
    deps = [
        "//programl/proto:programl_py",
        "//programl/util/py:init_app",
        "//programl/util/py:pbutil",
        "//programl/util/py:progress",
    ],
)

py_binary(
    name = "ggnn_test_one",
    srcs = ["ggnn_test_one.py"],
    deps = [
        "//models:base_graph_loader",
        "//models:batch_results",
        "//models/ggnn",
        "//programl/proto:programl_py",
        "//programl/util/py:init_app",
        "//programl/util/py:pbutil",
        "//tasks/dataflow",
        "//tasks/dataflow:ggnn_batch_builder",
        "//tasks/dataflow:vocabulary",
        "//tasks/dataflow/dataset:pathflag",
    ],
)

py_library(
    name = "graph_loader",
    srcs = ["graph_loader.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//models:base_graph_loader",
        "//programl/graph/format/py:cdfg",
        "//programl/proto:programl_py",
        "//programl/util/py:humanize",
        "//programl/util/py:pbutil",
    ],
)

py_library(
    name = "lstm_batch_builder",
    srcs = ["lstm_batch_builder.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//models:base_batch_builder",
        "//models:base_graph_loader",
        "//models:batch_data",
        "//models/lstm:lstm_batch",
        "//programl/graph/format/py:graph_serializer",
    ],
)

sh_binary(
    name = "overview",
    srcs = ["overview.sh"],
)

py_binary(
    name = "test_step_counts",
    srcs = ["test_step_counts.py"],
    deps = [
        "//programl/proto:programl_py",
        "//programl/util/py:init_app",
        "//programl/util/py:pbutil",
    ],
)

py_binary(
    name = "test_vocab",
    srcs = ["test_vocab.py"],
    deps = [
        ":vocabulary",
        "//programl/proto:programl_py",
        "//programl/third_party/ncc:vocabulary",
        "//programl/util/py:init_app",
        "//programl/util/py:pbutil",
        "//programl/util/py:progress",
        "//tasks/dataflow/dataset:pathflag",
    ],
)

py_binary(
    name = "train_ggnn",
    srcs = ["train_ggnn.py"],
    deps = [
        ":dataflow",
        ":ggnn",
        ":vocabulary",
        "//programl/util/py:init_app",
    ],
)

py_binary(
    name = "train_lstm",
    srcs = ["train_lstm.py"],
    deps = [
        ":dataflow",
        ":graph_loader",
        ":lstm_batch_builder",
        "//models:async_batch_builder",
        "//models:epoch_batch_iterator",
        "//models/lstm",
        "//programl/proto:programl_py",
        "//programl/third_party/ncc:vocabulary",
        "//programl/util/py:humanize",
        "//programl/util/py:init_app",
        "//programl/util/py:pbutil",
    ],
)

py_library(
    name = "vocabulary",
    srcs = ["vocabulary.py"],
    deps = [
        "//programl/util/py:humanize",
    ],
)

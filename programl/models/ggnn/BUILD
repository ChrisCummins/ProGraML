# Gated graph neural network.
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
    name = "aux_readout",
    srcs = ["aux_readout.py"],
    deps = [
    ],
)

py_library(
    name = "ggnn",
    srcs = ["ggnn.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":aux_readout",
        ":ggnn_batch",
        ":ggnn_model",
        ":ggnn_proper",
        ":loss",
        ":node_embeddings",
        ":readout",
        "//programl/graph/format/py:graph_tuple",
        "//programl/models:batch_data",
        "//programl/models:batch_results",
        "//programl/models:model",
        "//programl/proto:programl_py",
        "//programl/util/py:progress",
    ],
)

py_library(
    name = "ggnn_batch",
    srcs = ["ggnn_batch.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/graph/format/py:graph_tuple",
        "//programl/proto:programl_py",
    ],
)

py_library(
    name = "ggnn_layer",
    srcs = ["ggnn_layer.py"],
    deps = [
    ],
)

py_library(
    name = "ggnn_model",
    srcs = ["ggnn_model.py"],
    deps = [
        ":aux_readout",
        ":ggnn_proper",
        ":loss",
        ":metrics",
        ":node_embeddings",
    ],
)

py_library(
    name = "ggnn_proper",
    srcs = ["ggnn_proper.py"],
    deps = [
        ":ggnn_layer",
        ":messaging_layer",
        ":readout",
    ],
)

py_library(
    name = "linear_net",
    srcs = ["linear_net.py"],
    deps = [
    ],
)

py_library(
    name = "loss",
    srcs = ["loss.py"],
    deps = [
    ],
)

py_library(
    name = "messaging_layer",
    srcs = ["messaging_layer.py"],
    deps = [
        ":linear_net",
        ":position_embeddings",
    ],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    deps = [
    ],
)

py_library(
    name = "node_embeddings",
    srcs = ["node_embeddings.py"],
    deps = [
    ],
)

py_library(
    name = "position_embeddings",
    srcs = ["position_embeddings.py"],
    deps = [
    ],
)

py_library(
    name = "readout",
    srcs = ["readout.py"],
    deps = [
        ":linear_net",
    ],
)

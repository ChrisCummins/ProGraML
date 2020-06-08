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
        "//third_party/py/torch",
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
        "//labm8/py:app",
        "//labm8/py:progress",
        "//programl/graph/format/py:graph_tuple",
        "//programl/models:batch_data",
        "//programl/models:batch_results",
        "//programl/models:model",
        "//programl/proto:epoch_py",
        "//third_party/py/numpy",
        "//third_party/py/torch",
    ],
)

py_library(
    name = "ggnn_batch",
    srcs = ["ggnn_batch.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/graph/format/py:graph_tuple",
        "//programl/proto:program_graph_py",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "ggnn_layer",
    srcs = ["ggnn_layer.py"],
    deps = [
        "//third_party/py/torch",
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
        "//labm8/py:gpu_scheduler",
        "//third_party/py/torch",
    ],
)

py_library(
    name = "ggnn_proper",
    srcs = ["ggnn_proper.py"],
    deps = [
        ":ggnn_layer",
        ":messaging_layer",
        ":readout",
        "//third_party/py/torch",
    ],
)

py_library(
    name = "linear_net",
    srcs = ["linear_net.py"],
    deps = [
        "//third_party/py/torch",
    ],
)

py_library(
    name = "loss",
    srcs = ["loss.py"],
    deps = [
        "//third_party/py/torch",
    ],
)

py_library(
    name = "messaging_layer",
    srcs = ["messaging_layer.py"],
    deps = [
        ":linear_net",
        ":position_embeddings",
        "//third_party/py/torch",
    ],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    deps = [
        "//third_party/py/torch",
    ],
)

py_library(
    name = "node_embeddings",
    srcs = ["node_embeddings.py"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/torch",
    ],
)

py_library(
    name = "position_embeddings",
    srcs = ["position_embeddings.py"],
    deps = [
        "//third_party/py/torch",
    ],
)

py_library(
    name = "readout",
    srcs = ["readout.py"],
    deps = [
        ":linear_net",
        "//third_party/py/torch",
    ],
)

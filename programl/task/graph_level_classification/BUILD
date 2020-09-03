# Model implementations for graph-level classification tasks.
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
    name = "configs",
    srcs = ["configs.py"],
    deps = [":dataset"],
)

py_library(
    name = "dataloader",
    srcs = ["dataloader.py"],
    deps = [
        "//third_party/py/torch",
    ]
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        "//programl/util/py:deprecated",
        "//programl/graph/format/py:cdfg",
        "//programl/proto:program_graph_py",
        "//third_party/py/scikit_learn",
        "//third_party/py/torch",
        "//third_party/py/pandas",
        "//third_party/py/numpy",
        "//third_party/py/tqdm",
    ],
)

py_test(
    name="dataset_test",
    srcs=["dataset_test.py"],
    deps=[
        ":dataset",
        "//programl/test/py/plugins:reachability_dataflow_dataset",
        "//third_party/py/labm8",
    ],
)

py_library(
    name = "modeling",
    srcs = ["modeling.py"],
    deps = [
        "//third_party/py/torch",
    ]
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    deps = [
        "//third_party/py/torch",
        "//third_party/py/numpy",
        "//third_party/py/tqdm",
    ]
)

# LSTM models.
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
    name = "lstm",
    srcs = ["lstm.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":lstm_batch",
        "//labm8/py:app",
        "//labm8/py:progress",
        "//programl/models:batch_data",
        "//programl/models:model",
        "//programl/proto:epoch_py",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "lstm_batch",
    srcs = ["lstm_batch.py"],
    visibility = ["//visibility:public"],
    deps = [],
)

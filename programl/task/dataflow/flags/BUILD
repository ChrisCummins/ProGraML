# Flag files for configuring data flow model jobs.
#
# Example Usage:
#
#    bazel run -c opt //programl/task/dataflow:train_ggnn -- \
#       --flagfile=/path/to/flagfile.txt \
#       --analysis=reachability
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

filegroup(
    name = "flagfiles",
    srcs = [
        "cdfg.txt",
        "inst2vec.txt",
        "programl.txt",
    ],
    visibility = ["//programl/task/dataflow:__subpackages__"],
)

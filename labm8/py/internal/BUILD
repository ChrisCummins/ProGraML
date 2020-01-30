# Internal implementation libraries for labm8.
#
# Copyright 2014-2020 Chris Cummins <chrisc.101@gmail.com>.
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

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

py_library(
    name = "flags_parsers",
    srcs = ["flags_parsers.py"],
    visibility = ["//labm8/py:__subpackages__"],
    deps = [
        "//third_party/py/absl",
    ],
)

proto_library(
    name = "lockfile_pb",
    srcs = ["lockfile.proto"],
)

python_proto_library(
    name = "lockfile_pb_py",
    visibility = ["//labm8/py:__subpackages__"],
    deps = [":lockfile_pb"],
)

py_library(
    name = "labm8_logging",
    srcs = ["labm8_logging.py"],
    visibility = ["//labm8/py:__subpackages__"],
    deps = [
        "//third_party/py/absl",
    ],
)

proto_library(
    name = "logging_pb",
    srcs = ["logging.proto"],
)

python_proto_library(
    name = "logging_pb_py",
    visibility = ["//labm8/py:__subpackages__"],
    deps = [":logging_pb"],
)

py_library(
    name = "pytest_plugin",
    testonly = 1,
    srcs = ["pytest_plugin.py"],
    visibility = ["//labm8/py:__subpackages__"],
    deps = [
        "//labm8/py:app",
        "//third_party/py/pytest",
    ],
)

# A "python module" which exposes the workspace variables as module constants.
genrule(
    name = "workspace_status",
    outs = ["workspace_status.py"],
    cmd = ("sed -r 's/ (.*)/=\"\\1\"/' bazel-out/stable-status.txt > $@ &&" +
           "sed -r 's/ (.*)/=\"\\1\"/' bazel-out/volatile-status.txt >> $@"),
    stamp = 1,
    visibility = [
        "//labm8/py:__subpackages__",
        "//tools:__subpackages__",
    ],
)

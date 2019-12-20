# Copyright 2019 the ProGraML authors.
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

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")
load("//tools/bzl:exports.bzl", "exports_repo")

exports_repo(
    name = "export",
    github_repo = "ProGraML",
    move_file_mapping = {
        "deeplearning/ml4pl/bettercodehub.yml": ".bettercodehub.yml",
        "deeplearning/ml4pl/README.md": "README.md",
        "deeplearning/ml4pl/LICENSE": "LICENSE",
        "deeplearning/ml4pl/travis.yml": ".travis.yml",
    },
    targets = [
        "//deeplearning/ml4pl/...",
    ],
)

py_library(
    name = "filesystem_paths",
    srcs = ["filesystem_paths.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        "//labm8/py:app",
    ],
)

py_binary(
    name = "run_id",
    srcs = ["run_id.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        ":filesystem_paths",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:system",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "run_id_test",
    srcs = ["run_id_test.py"],
    deps = [
        ":run_id",
        "//labm8/py:sqlutil",
        "//labm8/py:test",
        "//third_party/py/sqlalchemy",
    ],
)

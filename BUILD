# Graph-based Machine Learning for Programming Languages.
# https://chriscummins.cc/ProGraML/
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

exports_files([
    ".bettercodehub.yml",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "version.txt",
    "WORKSPACE",
])

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "install",
    srcs = ["install.sh"],
    data = [
        "@labm8//labm8/sh:app",
        "//programl/cmd:analyze",
        "//programl/cmd:graph2cdfg",
        "//programl/cmd:clang2graph",
        "//programl/cmd:graph2dot",
        "//programl/cmd:graph2json",
        "//programl/cmd:llvm2graph",
        "//programl/cmd:pbq",
        "//programl/cmd:xla2graph",
    ] + select({
        "//:darwin": [
            "@clang-llvm-10.0.0-x86_64-apple-darwin//:libs",
        ],
        "//conditions:default": [
            "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:lib",
        ],
    }),
)

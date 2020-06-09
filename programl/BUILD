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
load("@cec_exports_repo//tools/bzl:exports_repo.bzl", "exports_repo")

exports_files([
    "README.md",
    "travis.yml",
    "bettercodehub.yml",
    "LICENSE",
])

exports_repo(
    name = "export",
    path_rename = {
        "programl/README.md": "README.md",
        "programl/LICENSE": "LICENSE",
        "programl/travis.yml": ".travis.yml",
        "programl/bettercodehub.yml": ".bettercodehub.yml",
    },
    paths = [
        "programl/**/*",
    ],
    remote = "git@github.com:ChrisCummins/ProGraML.git",
    tags = ["manual"],
    targets = [
        "//programl/...",
    ],
)

sh_binary(
    name = "install",
    srcs = ["install.sh"],
    data = [
        "//labm8/sh:app",
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
            "@llvm_mac//:libs",
        ],
        "//conditions:default": [
            "@llvm_linux//:libs",
        ],
    }),
)

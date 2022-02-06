# Labm8 is a Python library for manipulating quantitative experimental data.
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

load("@cec_exports_repo//tools/bzl:exports_repo.bzl", "exports_repo")
load("//tools/bzl:exports.bzl", "exports_pip")

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
    "README.md",
    "CNAME",
])

exports_repo(
    name = "export",
    path_rename = {
        "labm8/README.md": "README.md",
        "labm8/LICENSE": "LICENSE",
        "labm8/travis.yml": ".travis.yml",
    },
    paths = [
        "labm8/labm8.jpg",
    ],
    remote = "git@github.com:ChrisCummins/labm8.git",
    tags = ["manual"],
    targets = ["//labm8/..."],
)

exports_pip(
    name = "export_python_pip",
    package_name = "labm8",
    classifiers = [
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description = "Utility libraries for doing science",
    keywords = [
        "utility",
        "library",
        "bazel",
        "protobuf",
    ],
    license = "Apache License, Version 2.0",
    long_description_file = "//labm8:README.md",
    package_root = "//labm8",
)

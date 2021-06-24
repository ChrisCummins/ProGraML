#!/usr/bin/env python3
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

import distutils.util
import io

import setuptools

with io.open("version.txt", encoding="utf-8") as f:
    version = f.read().strip()
with open("README.md") as f:
    with io.open("README.md", encoding="utf-8") as f:
        long_description = f.read()
with open("programl/requirements.txt") as f:
    requirements = [ln.split("#")[0].rstrip() for ln in f.readlines()]

# When building a bdist_wheel we need to set the appropriate tags: this package
# includes compiled binaries, and does not include compiled python extensions.
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False

        def get_tag(self):
            python, abi, plat = _bdist_wheel.get_tag(self)
            python, abi = "py3", "none"
            return python, abi, plat


except ImportError:
    bdist_wheel = None

setuptools.setup(
    name="programl",
    version=version,
    description="A Graph-based Program Representation for Data Flow Analysis and Compiler Optimizations",
    author="Chris Cummins",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChrisCummins/ProGraML",
    license="Apache 2.0",
    packages=[
        "programl.ir.llvm",
        "programl.proto",
        "programl.third_party.inst2vec",
        "programl.third_party.tensorflow",
        "programl.util.py",
        "programl",
    ],
    package_dir={
        "": "bazel-bin/py_package.runfiles/programl",
    },
    package_data={
        "programl": [
            "bin/*",
            "ir/llvm/internal/*.pickle",
        ],
    },
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Compilers",
    ],
    cmdclass={"bdist_wheel": bdist_wheel},
    platforms=[distutils.util.get_platform()],
    zip_safe=False,
)

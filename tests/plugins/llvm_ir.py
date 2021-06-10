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
from pathlib import Path
from typing import Iterable, Tuple

import pytest

from programl.util.py.runfiles_path import runfiles_path

LLVM_IR = runfiles_path("tests/data/llvm_ir")


def EnumerateLlvmIrPaths() -> Iterable[Tuple[str, Path]]:
    """Enumerate a test set of LLVM IR file paths."""
    for path in LLVM_IR.iterdir():
        yield path.name, path


def EnumerateLlvmIrs(paths) -> Iterable[Tuple[str, str]]:
    """Enumerate a test set of LLVM IR file paths."""
    for name, path in paths:
        with open(str(path), "r") as f:
            yield name, f.read()


_LLVM_IR_PATHS = list(EnumerateLlvmIrPaths())
_LLVM_IRS = list(EnumerateLlvmIrs(_LLVM_IR_PATHS))


@pytest.fixture(
    scope="session", params=_LLVM_IR_PATHS, ids=[s[0] for s in _LLVM_IR_PATHS]
)
def llvm_ir_path(request) -> Path:
    """A test fixture which yields an LLVM-IR path."""
    return request.param[1]


@pytest.fixture(scope="session", params=_LLVM_IRS, ids=[s[0] for s in _LLVM_IRS])
def llvm_ir(request) -> str:
    """A test fixture which yields an LLVM-IR string."""
    return request.param[1]

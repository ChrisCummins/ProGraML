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


_LLVM_IR_PATHS = list(EnumerateLlvmIrPaths())


@pytest.fixture(
    scope="session", params=_LLVM_IR_PATHS, ids=[s[0] for s in _LLVM_IR_PATHS]
)
def llvm_ir_path(request) -> Path:
    """A test fixture which yields an LLVM-IR path."""
    return request.param[1]


@pytest.fixture(
    scope="session", params=_LLVM_IR_PATHS, ids=[s[0] for s in _LLVM_IR_PATHS]
)
def llvm_ir(request) -> str:
    """A test fixture which yields an LLVM-IR string."""
    with open(request.param[1], "r") as f:
        yield f.read()

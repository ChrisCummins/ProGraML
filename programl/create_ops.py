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
import subprocess
from typing import List, Optional

import google.protobuf.message

from programl.exceptions import GraphCreationError, UnsupportedCompiler
from programl.proto import ProgramGraph
from programl.third_party.tensorflow.xla_pb2 import HloProto
from programl.util.py.runfiles_path import runfiles_path

LLVM2GRAPH_BINARIES = {
    "10": str(runfiles_path("programl/bin/llvm2graph-10")),
    "3.8": str(runfiles_path("programl/bin/llvm2graph-10")),
}

CLANG2GRAPH_BINARIES = {
    "10": str(runfiles_path("programl/bin/clang2graph-10")),
}

XLA2GRAPH = str(runfiles_path("programl/bin/xla2graph"))


def _graph_from_subprocess(process, stdout, stderr):
    if process.returncode:
        try:
            raise GraphCreationError(stderr.decode("utf-8"))
        except UnicodeDecodeError as e:
            raise GraphCreationError("Unknown error in graph construction") from e

    try:
        graph = ProgramGraph()
        graph.ParseFromString(stdout)
        return graph
    except google.protobuf.message.DecodeError as e:
        raise GraphCreationError(str(e)) from e


# LLVM.


def from_llvm_ir(ir: str, timeout=300, version: str = "10") -> ProgramGraph:
    binary = LLVM2GRAPH_BINARIES.get(version)
    if not binary:
        raise UnsupportedCompiler(
            f"Unknown llvm version: {version}. "
            f"Supported versions: {sorted(LLVM2GRAPH_BINARIES.keys())}"
        )

    process = subprocess.Popen(
        [binary, "--stdout_fmt=pb"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        stdout, stderr = process.communicate(ir.encode("utf-8"), timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(str(e)) from e

    return _graph_from_subprocess(process, stdout, stderr)


def from_clang(args: str, version: str = "10", timeout=300) -> ProgramGraph:
    binary = CLANG2GRAPH_BINARIES.get(version)
    if not binary:
        raise UnsupportedCompiler(
            f"Unknown clang version: {version}. "
            f"Supported versions: {sorted(CLANG2GRAPH_BINARIES.keys())}"
        )

    process = subprocess.Popen(
        [binary] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(str(e)) from e

    return _graph_from_subprocess(process, stdout, stderr)


def from_cpp(
    src: str, copts: Optional[List[str]] = None, version: str = "10", timeout=300
) -> ProgramGraph:
    binary = CLANG2GRAPH_BINARIES.get(version)
    if not binary:
        raise UnsupportedCompiler(
            f"Unknown clang version: {version}. "
            f"Supported versions: {sorted(CLANG2GRAPH_BINARIES.keys())}"
        )

    process = subprocess.Popen(
        [binary, "-xc++", "-"] + (copts or []),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        stdout, stderr = process.communicate(src.encode("utf-8"), timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(str(e)) from e

    return _graph_from_subprocess(process, stdout, stderr)


# XLA.


def from_xla_hlo_proto(hlo: HloProto, timeout=300) -> ProgramGraph:
    process = subprocess.Popen(
        [XLA2GRAPH, "--stdin_fmt=pb", "--stdout_fmt=pb"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        stdout, stderr = process.communicate(hlo.SerializeToString(), timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(str(e)) from e

    return _graph_from_subprocess(process, stdout, stderr)

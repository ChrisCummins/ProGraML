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
"""Graph creation operations are used to construct Program Graphs from source
code or compiler intermediate representations (IRs).
"""
import subprocess
from typing import Iterable, List, Optional, Union

import google.protobuf.message

from programl.exceptions import GraphCreationError, UnsupportedCompiler
from programl.proto import ProgramGraph
from programl.third_party.tensorflow.xla_pb2 import HloProto
from programl.util.py.cc_system_includes import get_system_includes
from programl.util.py.executor import ExecutorLike, execute
from programl.util.py.runfiles_path import runfiles_path

LLVM2GRAPH_BINARIES = {
    "10": str(runfiles_path("programl/bin/llvm2graph-10")),
    "6": str(runfiles_path("programl/bin/llvm2graph-6")),
    "3.8": str(runfiles_path("programl/bin/llvm2graph-3.8")),
}

LLVM_VERSIONS = list(LLVM2GRAPH_BINARIES.keys())

CLANG2GRAPH_BINARIES = {
    "10": str(runfiles_path("programl/bin/clang2graph-10")),
}

CLANG_VERSIONS = list(CLANG2GRAPH_BINARIES.keys())

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


def from_cpp(
    srcs: Union[str, Iterable[str]],
    copts: Optional[List[str]] = None,
    system_includes: bool = True,
    language: str = "c++",
    version: str = "10",
    timeout=300,
    executor: Optional[ExecutorLike] = None,
    chunksize: Optional[int] = None,
) -> Union[ProgramGraph, Iterable[ProgramGraph]]:
    """Construct a Program Graph from a string of C/C++ code.

    This is a convenience function for generating graphs of simple single-file
    code snippets. For example:

        >>> programl.from_cpp(\"\"\" ... #include <stdio.h>
        ...
        ... int main() {
        ...   printf("Hello, ProGraML!");
        ...   return 0;
        ... }
        ... \"\"\")

    This is equivalent to invoking clang with input over stdin:

    .. code-block::

        cat <<EOF | clang -xc++ - -c -o -
        #include <stdio.h>

        int main() {
            printf("Hello, ProGraML!");
            return 0;
        }
        EOF

    For more control over the clang invocation, see :func:`from_clang`.

    :param srcs: A string of C / C++, or an iterable sequence of strings of C /
        C++.

    :param copts: A list of additional command line arguments to pass to clang.

    :param system_includes: Detect and pass :code:`-isystem` arguments to clang
        using the default search path of the system compiler. See
        :func:`get_system_includes()
        <programl.util.py.cc_system_includes.get_system_includes>` for details.

    :param language: The programming language of :code:`srcs`. Must be either
        :code:`c++` or :code:`c`.

    :param version: The version of clang to use. See
        :code:`programl.CLANG_VERSIONS` for a list of available versions.

    :param timeout: The maximum number of seconds to wait for an individual
        clang invocation before raising an error. If multiple :code:`srcs`
        inputs are provided, this timeout is per-input.

    :param executor: An executor object, with method :code:`submit(callable,
        *args, **kwargs)` and returning a Future-like object with methods
        :code:`done() -> bool` and :code:`result() -> float`. The executor role
        is to dispatch the execution of the jobs locally/on a cluster/with
        multithreading depending on the implementation. Eg:
        :code:`concurrent.futures.ThreadPoolExecutor`. Defaults to single
        threaded execution. This is only used when multiple inputs are given.

    :param chunksize: The number of inputs to read and process at a time. A
        larger chunksize improves parallelism but increases memory consumption
        as more inputs must be stored in memory.

    :return: If :code:`srcs` is singular, returns a single
        :code:`programl.ProgramGraph` instance. Else returns a generator over
        :code:`programl.ProgramGraph` instances.

    :raises UnsupportedCompiler: If the requested compiler version is not
        supported.

    :raises GraphCreationError: If compilation of the input fails.

    :raises TimeoutError: If the specified timeout is reached.
    """
    copts = copts or []
    binary = CLANG2GRAPH_BINARIES.get(version)
    if not binary:
        raise UnsupportedCompiler(
            f"Unknown clang version: {version}. "
            f"Supported versions: {sorted(CLANG2GRAPH_BINARIES.keys())}"
        )

    if system_includes:
        for directory in get_system_includes():
            copts += ["-isystem", str(directory)]

    def _run_one(src: str):
        process = subprocess.Popen(
            [binary, f"-x{language}", "-"] + copts,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            stdout, stderr = process.communicate(src.encode("utf-8"), timeout=timeout)
        except subprocess.TimeoutExpired as e:
            process.kill()
            raise TimeoutError(str(e)) from e

        return _graph_from_subprocess(process, stdout, stderr)

    if isinstance(srcs, str):
        return _run_one(srcs)
    return execute(_run_one, srcs, executor, chunksize)


def from_clang(
    args: Union[List[str], Iterable[List[str]]],
    system_includes: bool = True,
    version: str = "10",
    timeout=300,
    executor: Optional[ExecutorLike] = None,
    chunksize: Optional[int] = None,
) -> Union[ProgramGraph, Iterable[ProgramGraph]]:
    """Run clang and construct a Program Graph from the output.

    Example usage:

        >>> programl.from_clang(["/path/to/my/app.c", "-DMY_MACRO=3"])

    This is equivalent to invoking clang as:

    .. code-block::

        clang -c /path/to/my/app.c -DMY_MACRO=3

    Multiple inputs can be passed in a single invocation to be batched and
    processed in parallel. For example:

        >>> with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        ...     programl.from_clang(
        ...         ["a.cc", "-DMY_MACRO=3"],
        ...         ["b.cpp"],
        ...         ["c.c", "-O3", "-std=c99"],
        ...         executor=executor,
        ...     )


    :param args: A list of arguments to pass to clang, or an iterable sequence
        of arguments to pass to clang.

    :param system_includes: Detect and pass :code:`-isystem` arguments to clang
        using the default search path of the system compiler. See
        :func:`get_system_includes()
        <programl.util.py.cc_system_includes.get_system_includes>` for details.

    :param version: The version of clang to use. See
        :code:`programl.CLANG_VERSIONS` for a list of available versions.

    :param timeout: The maximum number of seconds to wait for an individual
        clang invocation before raising an error. If multiple inputs are
        provided, this timeout is per-input.

    :param executor: An executor object, with method :code:`submit(callable,
        *args, **kwargs)` and returning a Future-like object with methods
        :code:`done() -> bool` and :code:`result() -> float`. The executor role
        is to dispatch the execution of the jobs locally/on a cluster/with
        multithreading depending on the implementation. Eg:
        :code:`concurrent.futures.ThreadPoolExecutor`. Defaults to single
        threaded execution. This is only used when multiple inputs are given.

    :param chunksize: The number of inputs to read and process at a time. A
        larger chunksize improves parallelism but increases memory consumption
        as more inputs must be stored in memory.

    :return: If :code:`args` is a single list of arguments, returns a single
        :code:`programl.ProgramGraph` instance. Else returns a generator over
        :code:`programl.ProgramGraph` instances.

    :raises UnsupportedCompiler: If the requested compiler version is not
        supported.

    :raises GraphCreationError: If compilation of the input fails.

    :raises TimeoutError: If the specified timeout is reached.
    """
    binary = CLANG2GRAPH_BINARIES.get(version)
    if not binary:
        raise UnsupportedCompiler(
            f"Unknown clang version: {version}. "
            f"Supported versions: {sorted(CLANG2GRAPH_BINARIES.keys())}"
        )

    extra_copts = []
    if system_includes:
        for directory in get_system_includes():
            extra_copts += ["-isystem", str(directory)]

    def _run_one(one_args):
        process = subprocess.Popen(
            [binary] + one_args + extra_copts,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as e:
            process.kill()
            raise TimeoutError(str(e)) from e

        return _graph_from_subprocess(process, stdout, stderr)

    if isinstance(args, list) and args and isinstance(args[0], str):
        return _run_one(args)
    return execute(_run_one, args, executor, chunksize)


def from_llvm_ir(
    irs: Union[str, Iterable[str]],
    timeout=300,
    version: str = "10",
    executor: Optional[ExecutorLike] = None,
    chunksize: Optional[int] = None,
) -> Union[ProgramGraph, Iterable[ProgramGraph]]:
    """Construct a Program Graph from a string of LLVM-IR.

    This takes as input one or more LLVM-IR strings as generated by
    :code:`llvm-dis` from a bitcode file, or from :code:`clang` using arguments:
    :code:`-emit-llvm -S`.

    Example usage:

        >>> programl.from_llvm_ir(\"\"\"
        ... source_filename = "-"
        ... target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
        ... target triple = "x86_64-apple-macosx11.0.0"
        ...
        ... ; ...
        ... \"\"\")

    Multiple inputs can be passed in a single invocation to be batched and
    processed in parallel. For example:

        >>> with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        ...     graphs = programl.from_llvm_ir(llvm_ir_strings, executor=executor)

    :param irs: A string of LLVM-IR, or an iterable sequence of LLVM-IR strings.

    :param version: The version of LLVM to use. See
        :code:`programl.LLVM_VERSIONS` for a list of available versions.

    :param timeout: The maximum number of seconds to wait for an individual
        graph construction invocation before raising an error. If multiple
        inputs are provided, this timeout is per-input.

    :param executor: An executor object, with method :code:`submit(callable,
        *args, **kwargs)` and returning a Future-like object with methods
        :code:`done() -> bool` and :code:`result() -> float`. The executor role
        is to dispatch the execution of the jobs locally/on a cluster/with
        multithreading depending on the implementation. Eg:
        :code:`concurrent.futures.ThreadPoolExecutor`. Defaults to single
        threaded execution. This is only used when multiple inputs are given.

    :param chunksize: The number of inputs to read and process at a time. A
        larger chunksize improves parallelism but increases memory consumption
        as more inputs must be stored in memory.

    :return: If :code:`irs` is a single IR, returns a single
        :code:`programl.ProgramGraph` instance. Else returns a generator over
        :code:`programl.ProgramGraph` instances.

    :raises UnsupportedCompiler: If the requested LLVM version is not supported.

    :raises GraphCreationError: If graph construction fails.

    :raises TimeoutError: If the specified timeout is reached.
    """
    binary = LLVM2GRAPH_BINARIES.get(version)
    if not binary:
        raise UnsupportedCompiler(
            f"Unknown llvm version: {version}. "
            f"Supported versions: {sorted(LLVM2GRAPH_BINARIES.keys())}"
        )

    def _run_one(ir: str):
        process = subprocess.Popen(
            [binary, "--stdout_fmt=pb"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            stdout, stderr = process.communicate(ir.encode("utf-8"), timeout=timeout)
        except subprocess.TimeoutExpired as e:
            process.kill()
            raise TimeoutError(str(e)) from e

        return _graph_from_subprocess(process, stdout, stderr)

    if isinstance(irs, str):
        return _run_one(irs)
    return execute(_run_one, irs, executor, chunksize)


# XLA.


def from_xla_hlo_proto(
    hlos: Union[HloProto, Iterable[HloProto]],
    timeout=300,
    executor: Optional[ExecutorLike] = None,
    chunksize: Optional[int] = None,
) -> Union[ProgramGraph, Iterable[ProgramGraph]]:
    """Construct a Program Graph from an XLA HLO protocol buffer.

    :param hlos: A :code:`HloProto`, or an iterable sequence of :code:`HloProto`
        instances.

    :param timeout: The maximum number of seconds to wait for an individual
        graph construction invocation before raising an error. If multiple
        inputs are provided, this timeout is per-input.

    :param executor: An executor object, with method :code:`submit(callable,
        *args, **kwargs)` and returning a Future-like object with methods
        :code:`done() -> bool` and :code:`result() -> float`. The executor role
        is to dispatch the execution of the jobs locally/on a cluster/with
        multithreading depending on the implementation. Eg:
        :code:`concurrent.futures.ThreadPoolExecutor`. Defaults to single
        threaded execution. This is only used when multiple inputs are given.

    :param chunksize: The number of inputs to read and process at a time. A
        larger chunksize improves parallelism but increases memory consumption
        as more inputs must be stored in memory.

    :return: If :code:`hlos` is a single input, returns a single
        :code:`programl.ProgramGraph` instance. Else returns a generator over
        :code:`programl.ProgramGraph` instances.

    :raises GraphCreationError: If graph construction fails.

    :raises TimeoutError: If the specified timeout is reached.
    """

    def _run_one(hlo: HloProto) -> ProgramGraph:
        process = subprocess.Popen(
            [XLA2GRAPH, "--stdin_fmt=pb", "--stdout_fmt=pb"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            stdout, stderr = process.communicate(
                hlo.SerializeToString(), timeout=timeout
            )
        except subprocess.TimeoutExpired as e:
            process.kill()
            raise TimeoutError(str(e)) from e

        return _graph_from_subprocess(process, stdout, stderr)

    if isinstance(hlos, HloProto):
        return _run_one(hlos)
    return execute(_run_one, hlos, executor, chunksize)

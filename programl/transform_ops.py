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
"""The graph transform ops each take an iterable sequence of graphs and produce
an iterable sequence of transformed graphs.
"""
import json
import subprocess
from itertools import islice
from typing import Any, Dict, Iterable, Optional

import dgl
import networkx as nx
from dgl.heterograph import DGLHeteroGraph
from networkx.readwrite import json_graph as nx_json

from programl.exceptions import GraphTransformError
from programl.proto import ProgramGraph
from programl.util.py.executor import ExecutorLike, SequentialExecutor
from programl.util.py.runfiles_path import runfiles_path

GRAPH2DOT = str(runfiles_path("programl/bin/graph2dot"))
GRAPH2JSON = str(runfiles_path("programl/bin/graph2json"))

JsonDict = Dict[str, Any]


def _run_graph_transform_binary(
    binary: str,
    graphs: Iterable[ProgramGraph],
    timeout: int = 300,
    executor: Optional[ExecutorLike] = None,
    chunksize: int = 128,
) -> Iterable[bytes]:
    executor = executor or SequentialExecutor()

    def _process_one_graph(graph: ProgramGraph) -> JsonDict:
        process = subprocess.Popen(
            [binary, "--stdin_fmt=pb"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            stdout, stderr = process.communicate(
                graph.SerializeToString(), timeout=timeout
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(str(e)) from e

        if process.returncode:
            try:
                raise GraphTransformError(stderr.decode("utf-8"))
            except UnicodeDecodeError as e:
                raise GraphTransformError(
                    "Unknown error in graph transformation"
                ) from e

        return stdout

    graphs = iter(graphs)
    chunk = list(islice(graphs, chunksize))
    while chunk:
        futures = [executor.submit(_process_one_graph, graph) for graph in chunk]
        for future in futures:
            yield future.result()
        chunk = list(islice(graphs, chunksize))


def to_dot(
    graphs: Iterable[ProgramGraph],
    timeout: int = 300,
    executor: Optional[ExecutorLike] = None,
    chunksize: int = 128,
) -> Iterable[str]:
    """
    :param executor: An executor object, with method :code:`submit(callable,
        *args, **kwargs)` and returning a Future-like object with methods
        :code:`done() -> bool` and :code:`result() -> float`. The executor's
        role is to dispatch the execution of jobs with multithreading depending
        on the implementation. Eg:
        :code:`concurrent.futures.ThreadPoolExecutor`.
    """
    for dotgraph in _run_graph_transform_binary(
        GRAPH2DOT,
        graphs=graphs,
        timeout=timeout,
        executor=executor,
        chunksize=chunksize,
    ):
        yield dotgraph.decode("utf-8")


def to_json(
    graphs: Iterable[ProgramGraph],
    timeout: int = 300,
    executor: Optional[ExecutorLike] = None,
    chunksize: int = 128,
) -> Iterable[JsonDict]:
    """
    :param executor: An executor object, with method :code:`submit(callable,
        *args, **kwargs)` and returning a Future-like object with methods
        :code:`done() -> bool` and :code:`result() -> float`. The executor's
        role is to dispatch the execution of jobs with multithreading depending
        on the implementation. Eg:
        :code:`concurrent.futures.ThreadPoolExecutor`.
    """
    for json_data in _run_graph_transform_binary(
        GRAPH2JSON,
        graphs=graphs,
        timeout=timeout,
        executor=executor,
        chunksize=chunksize,
    ):
        try:
            yield json.loads(json_data)
        except json.JSONDecodeError as e:
            raise GraphTransformError(str(e)) from e


def to_networkx(
    graphs: Iterable[ProgramGraph],
    timeout: int = 300,
    executor: Optional[ExecutorLike] = None,
    chunksize: int = 128,
) -> Iterable[nx.MultiDiGraph]:
    for json_data in to_json(
        graphs, timeout=timeout, executor=executor, chunksize=chunksize
    ):
        yield nx_json.node_link_graph(json_data, multigraph=True, directed=True)


def to_dgl(
    graphs: Iterable[ProgramGraph],
    timeout: int = 300,
    executor: Optional[ExecutorLike] = None,
    chunksize: int = 128,
) -> Iterable[DGLHeteroGraph]:
    for nx_graph in to_networkx(
        graphs, timeout=timeout, executor=executor, chunksize=chunksize
    ):
        yield dgl.DGLGraph(nx_graph)

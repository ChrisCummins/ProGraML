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
"""The graph transform ops are used to modify or convert Program Graphs to
another representation.
"""
import json
import subprocess
from typing import Any, Dict, Iterable, Optional, Union

import dgl
import networkx as nx
from dgl.heterograph import DGLHeteroGraph
from networkx.readwrite import json_graph as nx_json

from programl.exceptions import GraphTransformError
from programl.proto import ProgramGraph
from programl.util.py.executor import ExecutorLike, execute
from programl.util.py.runfiles_path import runfiles_path

GRAPH2DOT = str(runfiles_path("programl/bin/graph2dot"))
GRAPH2JSON = str(runfiles_path("programl/bin/graph2json"))

JsonDict = Dict[str, Any]


def _run_graph_transform_binary(
    binary: str,
    graph: ProgramGraph,
    timeout: int = 300,
) -> Iterable[bytes]:
    process = subprocess.Popen(
        [binary, "--stdin_fmt=pb"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        stdout, stderr = process.communicate(graph.SerializeToString(), timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(str(e)) from e

    if process.returncode:
        try:
            raise GraphTransformError(stderr.decode("utf-8"))
        except UnicodeDecodeError as e:
            raise GraphTransformError("Unknown error in graph transformation") from e

    return stdout


def to_json(
    graphs: Union[ProgramGraph, Iterable[ProgramGraph]],
    timeout: int = 300,
    executor: Optional[ExecutorLike] = None,
    chunksize: Optional[int] = None,
) -> Union[JsonDict, Iterable[JsonDict]]:
    """Convert one or more Program Graphs to JSON node-link data.

    :param graphs: A Program Graph, or a sequence of Program Graphs.

    :param timeout: The maximum number of seconds to wait for an individual
        graph conversion before raising an error. If multiple inputs are
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
        as more inputs must be stored in memory. This is only used when multiple
        inputs are given.

    :return: If a single input is provided, return a single JSON dictionary.
        Else returns an iterable sequence of JSON dictionaries.

    :raises GraphTransformError: If graph conversion fails.

    :raises TimeoutError: If the specified timeout is reached.
    """

    def _run_one(graph: ProgramGraph):
        try:
            return json.loads(
                _run_graph_transform_binary(
                    GRAPH2JSON,
                    graph,
                    timeout,
                )
            )
        except json.JSONDecodeError as e:
            raise GraphTransformError(str(e)) from e

    if isinstance(graphs, ProgramGraph):
        return _run_one(graphs)
    return execute(_run_one, graphs, executor, chunksize)


def to_networkx(
    graphs: Union[ProgramGraph, Iterable[ProgramGraph]],
    timeout: int = 300,
    executor: Optional[ExecutorLike] = None,
    chunksize: Optional[int] = None,
) -> Union[nx.MultiDiGraph, Iterable[nx.MultiDiGraph]]:
    """Convert one or more Program Graphs to `NetworkX MultiDiGraphs
    <https://networkx.org/documentation/stable/reference/classes/multidigraph.html>`_.

    :param graphs: A Program Graph, or a sequence of Program Graphs.

    :param timeout: The maximum number of seconds to wait for an individual
        graph conversion before raising an error. If multiple inputs are
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
        as more inputs must be stored in memory. This is only used when multiple
        inputs are given.

    :return: If a single input is provided, return a single :code:`nx.MultiDiGraph`.
        Else returns an iterable sequence of :code:`nx.MultiDiGraph` instances.

    :raises GraphTransformError: If graph conversion fails.

    :raises TimeoutError: If the specified timeout is reached.
    """

    def _run_one(json_data):
        return nx_json.node_link_graph(json_data, multigraph=True, directed=True)

    if isinstance(graphs, ProgramGraph):
        return _run_one(to_json(graphs, timeout=timeout))
    return execute(
        _run_one,
        to_json(graphs, timeout=timeout, executor=executor, chunksize=chunksize),
        executor,
        chunksize,
    )


def to_dgl(
    graphs: Union[ProgramGraph, Iterable[ProgramGraph]],
    timeout: int = 300,
    executor: Optional[ExecutorLike] = None,
    chunksize: Optional[int] = None,
) -> Union[DGLHeteroGraph, Iterable[DGLHeteroGraph]]:
    """Convert one or more Program Graphs to `DGLGraphs
    <https://docs.dgl.ai/en/latest/api/python/dgl.DGLGraph.html#dgl.DGLGraph>`_.

    :param graphs: A Program Graph, or a sequence of Program Graphs.

    :param timeout: The maximum number of seconds to wait for an individual
        graph conversion before raising an error. If multiple inputs are
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
        as more inputs must be stored in memory. This is only used when multiple
        inputs are given.

    :return: If a single input is provided, return a single
        :code:`dgl.DGLGraph`. Else returns an iterable sequence of
        :code:`dgl.DGLGraph` instances.

    :raises GraphTransformError: If graph conversion fails.

    :raises TimeoutError: If the specified timeout is reached.
    """

    def _run_one(nx_graph):
        return dgl.DGLGraph(nx_graph)

    if isinstance(graphs, ProgramGraph):
        return _run_one(to_networkx(graphs))
    return execute(
        _run_one,
        to_networkx(graphs, timeout=timeout, executor=executor, chunksize=chunksize),
        executor,
        chunksize,
    )


def to_dot(
    graphs: Union[ProgramGraph, Iterable[ProgramGraph]],
    timeout: int = 300,
    executor: Optional[ExecutorLike] = None,
    chunksize: Optional[int] = None,
) -> Union[str, Iterable[str]]:
    """Convert one or more Program Graphs to DOT Graph Description Language.

    This produces a DOT source string representing the input graph. This can
    then be rendered using the graphviz command line tools, or parsed using
    `pydot <https://pypi.org/project/pydot/>`_.

    :param graphs: A Program Graph, or a sequence of Program Graphs.

    :param timeout: The maximum number of seconds to wait for an individual
        graph conversion before raising an error. If multiple inputs are
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
        as more inputs must be stored in memory. This is only used when multiple
        inputs are given.

    :return: A graphviz dot string when a single input is provided, else an
        iterable sequence of graphviz dot strings.

    :raises GraphTransformError: If graph conversion fails.

    :raises TimeoutError: If the specified timeout is reached.
    """

    def _run_one(graph: ProgramGraph) -> str:
        return _run_graph_transform_binary(GRAPH2DOT, graph, timeout).decode("utf-8")

    if isinstance(graphs, ProgramGraph):
        return _run_one(graphs)
    return execute(_run_one, graphs, executor, chunksize)

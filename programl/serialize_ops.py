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
"""Graph serialization ops are used for storing or transferring Program
Graphs.
"""
import gzip
from pathlib import Path
from typing import Iterable, List, Optional

import google.protobuf.message
import google.protobuf.text_format

from programl.exceptions import GraphCreationError
from programl.proto import ProgramGraph, ProgramGraphList


def save_graphs(
    path: Path, graphs: Iterable[ProgramGraph], compression: Optional[str] = "gz"
) -> None:
    """Save a sequence of program graphs to a file.

    :param path: The file to write.

    :param graphs: A sequence of Program Graphs.

    :param compression: Either :code:`gz` for GZip compression (the default), or
        :code:`None` for no compression. Compression increases the cost of
        serializing and deserializing but can greatly reduce the size of the
        serialized graphs.

    :raises TypeError: If an unsupported :code:`compression` is given.
    """
    with open(path, "wb") as f:
        f.write(to_bytes(graphs, compression=compression))


def load_graphs(
    path: Path, idx_list: Optional[List[int]] = None, compression: Optional[str] = "gz"
) -> List[ProgramGraph]:
    """Load program graphs from a file.

    :param path: The file to read from.

    :param idx_list: A zero-based list of graph indices to return. If not
        provided, all graphs are loaded.

    :param compression: Either :code:`gz` for GZip compression (the default), or
        :code:`None` for no compression. Compression increases the cost of
        serializing and deserializing but can greatly reduce the size of the
        serialized graphs.

    :return: A sequence of Program Graphs.

    :raises TypeError: If an unsupported :code:`compression` is given.

    :raise GraphCreationError: If deserialization fails.
    """
    with open(path, "rb") as f:
        return from_bytes(f.read(), idx_list=idx_list, compression=compression)


def to_bytes(
    graphs: Iterable[ProgramGraph], compression: Optional[str] = "gz"
) -> bytes:
    """Serialize a sequence of Program Graphs to a byte array.

    :param graphs: A sequence of Program Graphs.

    :param compression: Either :code:`gz` for GZip compression (the default), or
        :code:`None` for no compression. Compression increases the cost of
        serializing and deserializing but can greatly reduce the size of the
        serialized graphs.

    :return: The serialized program graphs.
    """
    compressors = {
        "gz": gzip.compress,
        None: lambda d: d,
    }
    if compression not in compressors:
        compressors = ", ".join(sorted(str(x) for x in compressors))
        raise TypeError(
            f"Invalid compression argument: {compression}. "
            f"Supported compressions: {compressors}"
        )
    compress = compressors[compression]

    return compress(ProgramGraphList(graph=list(graphs)).SerializeToString())


def from_bytes(
    data: bytes, idx_list: Optional[List[int]] = None, compression: Optional[str] = "gz"
) -> List[ProgramGraph]:
    """Deserialize Program Graphs from a byte array.

    :param data: The serialized Program Graphs.

    :param idx_list: A zero-based list of graph indices to return. If not
        provided, all graphs are returned.

    :param compression: Either :code:`gz` for GZip compression (the default), or
        :code:`None` for no compression. Compression increases the cost of
        serializing and deserializing but can greatly reduce the size of the
        serialized graphs.

    :return: A list of Program Graphs.

    :raise GraphCreationError: If deserialization fails.
    """
    decompressors = {
        "gz": gzip.decompress,
        None: lambda d: d,
    }
    if compression not in decompressors:
        decompressors = ", ".join(sorted(str(x) for x in decompressors))
        raise TypeError(
            f"Invalid compression argument: {compression}. "
            f"Supported compressions: {decompressors}"
        )
    decompress = decompressors[compression]

    graph_list = ProgramGraphList()
    try:
        graph_list.ParseFromString(decompress(data))
    except (gzip.BadGzipFile, google.protobuf.message.DecodeError) as e:
        raise GraphCreationError(str(e)) from e

    if idx_list:
        return [graph_list.graph[i] for i in idx_list]
    return list(graph_list.graph)


def to_string(graphs: Iterable[ProgramGraph]) -> str:
    """Serialize a sequence of Program Graphs to a human-readable string.

    The generated string has a JSON-like syntax that is designed for human
    readability. This is the least compact form of serialization.

    :param graphs: A sequence of Program Graphs.

    :return: The serialized program graphs.
    """
    return str(ProgramGraphList(graph=list(graphs)))


def from_string(
    string: str, idx_list: Optional[List[int]] = None
) -> List[ProgramGraph]:
    """Deserialize Program Graphs from a human-readable string.

    :param data: The serialized Program Graphs.

    :param idx_list: A zero-based list of graph indices to return. If not
        provided, all graphs are returned.

    :return: A list of Program Graphs.

    :raise GraphCreationError: If deserialization fails.
    """
    graph_list = ProgramGraphList()
    try:
        google.protobuf.text_format.Merge(string, graph_list)
    except google.protobuf.text_format.ParseError as e:
        raise GraphCreationError(str(e)) from e

    if idx_list:
        return [graph_list.graph[i] for i in idx_list]
    return list(graph_list.graph)

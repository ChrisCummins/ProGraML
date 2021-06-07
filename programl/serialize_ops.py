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
    compressors = {
        "gz": gzip.compress,
        None: lambda d: d,
    }
    if compression not in compressors:
        raise TypeError(
            "Invalid compression argument: {compression}. "
            f"Supported compressions: {sorted(compressors.keys())}"
        )
    compress = compressors[compression]
    with open(path, "wb") as f:
        f.write(compress(to_bytes(graphs)))


def load_graphs(
    path: Path, idx_list: Optional[List[int]] = None, compression: Optional[str] = "gz"
) -> List[ProgramGraph]:
    decompressors = {
        "gz": gzip.decompress,
        None: lambda d: d,
    }
    if compression not in decompressors:
        raise TypeError(
            "Invalid compression argument: {decompression}. "
            f"Supported compressions: {sorted(decompressors.keys())}"
        )
    decompress = decompressors[compression]
    with open(path, "rb") as f:
        return from_bytes(decompress(f.read()))


def to_bytes(graphs: Iterable[ProgramGraph]) -> bytes:
    return ProgramGraphList(graph=list(graphs).SerializeToString())


def from_bytes(data: bytes, idx_list: Optional[List[int]] = None) -> List[ProgramGraph]:
    graph_list = ProgramGraphList()
    try:
        graph_list.ParseFromString(data)
    except google.protobuf.message.DecodeError as e:
        raise GraphCreationError(str(e)) from e
    if idx_list:
        return [graph_list.graph[i] for i in idx_list]
    else:
        return list(graph_list.graph)


def to_string(graphs: Iterable[ProgramGraph]) -> str:
    return str(ProgramGraphList(graph=list(graphs).SerializeToString()))


def from_string(
    string: str, idx_list: Optional[List[int]] = None
) -> List[ProgramGraph]:
    graph_list = ProgramGraphList()
    try:
        google.protobuf.text_format.Merge(string, graph_list)
    except google.protobuf.text_format.ParseError as e:
        raise GraphCreationError(str(e)) from e

    if idx_list:
        return [graph_list.graph[i] for i in idx_list]
    else:
        return list(graph_list.graph)

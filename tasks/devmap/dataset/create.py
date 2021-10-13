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
"""Prepare the CPU/GPU OpenCL device-mapping dataset."""
import hashlib
import io
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
from absl import app, flags

import programl as pg
from programl.util.py import pbutil
from tasks.dataflow.dataset import pathflag

FLAGS = flags.FLAGS


def cachedir() -> Path:
    """Return the path of the cache directory."""
    if os.environ.get("TEST_TMPDIR"):
        return Path(os.environ["TEST_TMPDIR"])
    else:
        return Path("~/.cache/programl").expanduser()


def download(url: str, checksum: str) -> bytes:
    """Download from a URL and return its contents."""
    cachepath = cachedir() / f"{checksum}.data"
    if cachepath.is_file():
        with open(cachepath, "rb") as f:
            content = f.read()
    else:
        print("downloading", url, "...")
        content = requests.get(url).content
        cachepath.parent.mkdir(parents=True, exist_ok=True)
        with open(cachepath, "wb") as f:
            f.write(content)

    sha256 = hashlib.sha256()
    sha256.update(content)
    actual_checksum = sha256.hexdigest()
    if actual_checksum != checksum:
        raise ValueError(
            f"Checksum mismatch of downloaded file {url}. "
            f"Expected: {checksum}. Actual: {actual_checksum}"
        )
    return content


def download_csv(url: str, checksum: str) -> pd.DataFrame:
    """Download and return a CSV file as a pandas data frame."""
    return pd.read_csv(io.StringIO(download(url, checksum).decode("utf-8")))


def reshape_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and reshape the useful bits of the dataframe."""
    names = [
        f"{benchmark}-{dataset}"
        for benchmark, dataset in df[["benchmark", "dataset"]].values
    ]
    return pd.DataFrame(
        {
            "name": names,
            "transfer_bytes": df["transfer"],
            "transfer_bytes_log1p": np.log1p(df["transfer"]),
            "wgsize": df["wgsize"],
            "wgsize_log1p": np.log1p(df["transfer"]),
            "label": df["runtime_gpu"] < df["runtime_cpu"],
        }
    )


def name2ncc_path(name: str, src_dir: Path, extension: str):
    """Resolve a NCC data archive path from a kernel name."""
    path = src_dir / f"{name}{extension}"
    if path.is_file():
        return path

    # Some of the benchmark sources are dataset dependent. This is reflected by
    # the dataset name being concatenated to the path.
    name_components = name.split("-")

    new_name = "-".join(name_components[:-1])
    path = src_dir / f"{new_name}{extension}"
    if path.is_file():
        return path

    new_name = "-".join(name_components[:-1]) + "_" + name_components[-1]
    path = src_dir / f"{new_name}{extension}"
    if path.is_file():
        return path

    raise FileNotFoundError(f"No OpenCL source found for {name}")


def dump_src(path: Path, df: pd.DataFrame, ncc_dir: Path):
    """Dump the OpenCL source files."""
    for name in df["name"].values:
        try:
            src = name2ncc_path(name, ncc_dir / "kernels_cl", ".cl")
            dst = path / "src" / f"{name}.cl"
            shutil.copyfile(src, dst)
        except FileNotFoundError:
            # Not all kernels correspond to OpenCL files. This is fine.
            pass


def dump_ir(path: Path, df: pd.DataFrame, ncc_dir: Path):
    """Dump the LLVM-IR files."""
    for name in df["name"].values:
        src = name2ncc_path(name, ncc_dir / "kernels_ir", ".ll")
        dst = path / "ir" / f"{name}.ll"
        shutil.copyfile(src, dst)


def build_graphs(df: pd.DataFrame, ir_dir: Path, graph_dir: Path):
    """Build ProgramGraphs from LLVM-IR and features."""
    for _, row in df.iterrows():
        with open(ir_dir / f"{row['name']}.ll") as f:
            ir = f.read()
        graph = pg.from_llvm_ir(ir)
        graph.features.feature["devmap_label"].int64_list.value[:] = [row["label"]]
        graph.features.feature["wgsize"].int64_list.value[:] = [row["wgsize"]]
        graph.features.feature["transfer_bytes"].int64_list.value[:] = [
            row["transfer_bytes"]
        ]
        graph.features.feature["wgsize_log1p"].float_list.value[:] = [
            row["wgsize_log1p"]
        ]
        graph.features.feature["transfer_bytes_log1p"].float_list.value[:] = [
            row["transfer_bytes_log1p"]
        ]
        pbutil.ToFile(
            graph, graph_dir / f"{row['name']}.ProgramGraph.pb", exist_ok=False
        )


def create_devmap_dataset(path: Path):
    """Create the devmap dataset."""
    # First create the output directories. Fail if they already exist.
    (path / "graphs_amd").mkdir(parents=True)
    (path / "graphs_nvidia").mkdir()
    (path / "ir").mkdir()
    (path / "src").mkdir()

    amd = download_csv(
        url="http://raw.githubusercontent.com/ChrisCummins/phd/65643fa5ad6769ce4678535cd2f9f37b6a467c45/datasets/opencl/device_mapping/amd.csv",
        checksum="0076271192aa9a0a7c21aa9a637e34cd4460f8e21e756215dd23ffb2ae62dc62",
    )
    nvidia = download_csv(
        url="http://raw.githubusercontent.com/ChrisCummins/phd/65643fa5ad6769ce4678535cd2f9f37b6a467c45/datasets/opencl/device_mapping/nvidia.csv",
        checksum="095c1ccef333e0a65e0e70b3ebde0aef851b61528ec46496a5d1687905abd099",
    )
    opencl_ir_zip = download(
        # Upstream URL: https://github.com/spcl/ncc/tree/master/task
        url="https://www.dropbox.com/s/j5ck80fsbuebf5g/devmap_data.zip?dl=1",
        checksum="3c840f84936a83e329c7a94d011c45ddfcfce8bdbb1a9b1904123e83851913d5",
    )

    amd = reshape_df(amd)
    nvidia = reshape_df(nvidia)

    with tempfile.TemporaryDirectory() as tmpdir:
        with ZipFile(io.BytesIO(opencl_ir_zip), "r") as f:
            f.extractall(tmpdir)
            dump_src(path, amd, Path(tmpdir))
            dump_ir(path, amd, Path(tmpdir))

    build_graphs(amd, path / "ir", path / "graphs_amd")
    build_graphs(nvidia, path / "ir", path / "graphs_nvidia")


def main():
    """Main entry point."""
    create_devmap_dataset(Path(pathflag.path()))


if __name__ == "__main__":
    app.Run(main)

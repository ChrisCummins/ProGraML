"""Data loader for GGNN."""
import csv
import enum
import math
import pickle
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
from torch_geometric.data import Data, InMemoryDataset

from programl.graph.format.py import cdfg as cdfg_format
from programl.proto.program_graph_pb2 import ProgramGraph
from programl.util.py.deprecated import deprecated


def load(file: Union[str, Path], cdfg: bool = False) -> ProgramGraph:
    """Read a ProgramGraph protocol buffer from file.

    Args:
        file: The path of the ProgramGraph protocol buffer to load.
        cdfg: If true, convert the graph to CDFG during load.

    Returns:
        A ProgramGraph proto instance.
    """
    graph = ProgramGraph()
    with open(file, "rb") as f:
        proto = f.read()
    graph.ParseFromString(proto)

    if cdfg:
        # Load CDFG and copy over graph-level features.
        cdfg_graph = cdfg_format.FromProgramGraphFile(Path(file))
        cdfg_graph.features.CopyFrom(graph.features)
        graph = cdfg_graph

    return graph


def load_vocabulary(path: Path):
    """Read the vocabulary file used in the dataflow experiments."""
    vocab = {}
    with open(path) as f:
        vocab_file = csv.reader(f.readlines(), delimiter="\t")
        for i, row in enumerate(vocab_file, start=-1):
            if i == -1:  # Skip the header.
                continue
            (_, _, _, text) = row
            vocab[text] = i

    return vocab


class AblationVocab(enum.IntEnum):
    # No ablation - use the full vocabulary (default).
    NONE = 0
    # Ignore the vocabulary - every node has an x value of 0.
    NO_VOCAB = 1
    # Use a 3-element vocabulary based on the node type:
    #    0 - Instruction node
    #    1 - Variable node
    #    2 - Constant node
    NODE_TYPE_ONLY = 2


def filename(
    split: str, cdfg: bool = False, ablation_vocab: AblationVocab = AblationVocab.NONE
) -> str:
    """Generate the name for a data file.

    Args:
        split: The name of the split.
        cdfg: Whether using CDFG representation.
        ablate_vocab: The ablation vocab type.

    Returns:
        A file name which uniquely identifies this combination of
        split/cdfg/ablation.
    """
    name = str(split)
    if cdfg:
        name = f"{name}_cdfg"
    if ablation_vocab != AblationVocab.NONE:
        # transform if ablation_vocab was passed as int.
        if type(ablation_vocab) == int:
            ablation_vocab = AblationVocab(ablation_vocab)
        name = f"{name}_{ablation_vocab.name.lower()}"
    return f"{name}_data.pt"


def nx2data(
    graph: ProgramGraph,
    vocabulary: Dict[str, int],
    y_feature_name: Optional[str] = None,
    ignore_profile_info=True,
    ablate_vocab=AblationVocab.NONE,
):
    """Converts a program graph protocol buffer to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        graph           A program graph protocol buffer.
        vocabulary      A map from node text to vocabulary indices.
        y_feature_name  The name of the graph-level feature to use as class label.
        ablate_vocab    Whether to use an ablation vocabulary.
    """

    # collect edge_index
    edge_tuples = [(edge.source, edge.target) for edge in graph.edge]
    edge_index = torch.tensor(edge_tuples).t().contiguous()

    # collect edge_attr
    positions = torch.tensor([edge.position for edge in graph.edge])
    flows = torch.tensor([int(edge.flow) for edge in graph.edge])

    edge_attr = torch.cat([flows, positions]).view(2, -1).t().contiguous()

    # collect x
    if ablate_vocab == AblationVocab.NONE:
        vocabulary_indices = vocab_ids = [
            vocabulary.get(node.text, len(vocabulary)) for node in graph.node
        ]
    elif ablate_vocab == AblationVocab.NO_VOCAB:
        vocabulary_indices = [0] * len(graph.node)
    elif ablate_vocab == AblationVocab.NODE_TYPE_ONLY:
        vocabulary_indices = [int(node.type) for node in graph.node]
    else:
        raise NotImplementedError("unreachable")

    xs = torch.tensor(vocabulary_indices)
    types = torch.tensor([int(node.type) for node in graph.node])

    x = torch.cat([xs, types]).view(2, -1).t().contiguous()

    assert (
        edge_attr.size()[0] == edge_index.size()[1]
    ), f"edge_attr={edge_attr.size()} size mismatch with edge_index={edge_index.size()}"

    data_dict = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }

    # maybe collect these data too
    if y_feature_name is not None:
        y = torch.tensor(
            graph.features.feature[y_feature_name].int64_list.value[0]
        ).view(
            1
        )  # <1>
        if y_feature_name == "poj104_label":
            y -= 1
        data_dict["y"] = y

    # branch prediction / profile info specific
    if not ignore_profile_info:
        raise NotImplementedError(
            "profile info is not supported with the new nx2data (from programgraph) adaptation."
        )
        profile_info = []
        for i, node_data in nx_graph.nodes(data=True):
            # default to -1, -1, -1 if not all profile info is given.
            if not (
                node_data.get("llvm_profile_true_weight") is not None
                and node_data.get("llvm_profile_false_weight") is not None
                and node_data.get("llvm_profile_total_weight") is not None
            ):
                mask = 0
                true_weight = -1
                false_weight = -1
                total_weight = -1
            else:
                mask = 1
                true_weight = node_data["llvm_profile_true_weight"]
                false_weight = node_data["llvm_profile_false_weight"]
                total_weight = node_data["llvm_profile_total_weight"]

            profile_info.append([mask, true_weight, false_weight, total_weight])

        data_dict["profile_info"] = torch.tensor(profile_info)

    # make Data
    data = Data(**data_dict)

    return data


class BranchPredictionDataset(InMemoryDataset):
    @deprecated
    def __init__(
        self,
        root="deeplearning/ml4pl/poj104/branch_prediction_data",
        split="train",
        transform=None,
        pre_transform=None,
        train_subset=[0, 100],
        train_subset_seed=0,
    ):
        """
        Args:
            train_subset: [start_percentile, stop_percentile)    default [0,100).
                            sample a random (but fixed) train set of data in slice by percent, with given seed.
            train_subset_seed: seed for the train_subset fixed random permutation.
        """
        self.split = split
        self.train_subset = train_subset
        self.train_subset_seed = train_subset_seed
        super().__init__(root, transform, pre_transform)

        assert split in [
            "train"
        ], "The BranchPrediction dataset only has a 'train' split. use train_subset=[0,x] and [x, 100] for training and testing."
        self.data, self.slices = torch.load(self.processed_paths[0])
        pass

    @property
    def raw_file_names(self):
        """A list of files that need to be found in the raw_dir in order to skip the download"""
        return []  # not implemented here

    @property
    def processed_file_names(self):
        """A list of files in the processed_dir which needs to be found in order to skip the processing."""
        base = f"{self.split}_data.pt"

        if tuple(self.train_subset) == (0, 100) or self.split in ["val", "test"]:
            return [base]
        else:
            assert self.split == "train"
            return [
                f"{self.split}_data_subset_{self.train_subset[0]}_{self.train_subset[1]}_seed_{self.train_subset_seed}.pt"
            ]

    def download(self):
        """Download raw data to `self.raw_dir`"""
        pass  # not implemented

    def _save_train_subset(self):
        """saves a train_subset of self to file.
        Percentile slice is taken according to self.train_subset
        with a fixed random permutation with self.train_subset_seed.
        """
        import numpy as np

        perm = np.random.RandomState(self.train_subset_seed).permutation(len(self))

        # take slice of perm according to self.train_subset
        start = np.math.floor(len(self) / 100 * self.train_subset[0])
        stop = np.math.floor(len(self) / 100 * self.train_subset[1])
        perm = perm[start:stop]
        print(f"Fixed permutation starts with: {perm[:min(30, len(perm))]}")

        dataset = self.__indexing__(perm)

        data, slices = dataset.data, dataset.slices
        torch.save((data, slices), self.processed_paths[0])
        return

    def return_cross_validation_splits(self, split):
        assert self.train_subset == [
            0,
            100,
        ], "Do cross-validation on the whole dataset!"
        # num_samples = len(self)
        # perm = np.random.RandomState(self.train_subset_seed).permutation(len(self))

        # 10-fold cross-validation
        n_splits = 10
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        (train_index, test_index) = list(kf.split(range(len(self))))[split]
        train_data = self.__indexing__(train_index)
        test_data = self.__indexing__(test_index)
        return train_data, test_data

    def filter_max_num_nodes(self, max_num_nodes):
        idx = []
        for i, d in enumerate(self):
            if d.num_nodes <= max_num_nodes:
                idx.append(i)
        dataset = self.__indexing__(idx)
        print(
            f"Filtering out graphs larger than {max_num_nodes} yields a dataset with {len(dataset)}/{len(self)} samples remaining."
        )
        return dataset

    def process(self):
        """Processes raw data and saves it into the `processed_dir`.
        New implementation:
            Here specifically it will collect all '*.ll.pickle' files recursively from subdirectories of `root`
            and process the loaded nx graphs to Data.
        Old implementation:
            Instead of looking for .ll.pickle (nx graphs), we directly look for '*.data.p' files.
        """
        # check if we need to create the full dataset:
        full_dataset = Path(self.processed_dir) / f"{self.split}_data.pt"
        if full_dataset.is_file():
            assert self.split == "train", "here shouldnt be reachable."
            print(
                f"Full dataset found. Generating train_subset={self.train_subset} with seed={self.train_subset_seed}"
            )
            # just get the split and save it
            self.data, self.slices = torch.load(full_dataset)
            self._save_train_subset()
            print(
                f"Saved train_subset={self.train_subset} with seed={self.train_subset_seed} to disk."
            )
            return

        # ~~~~~ we need to create the full dataset ~~~~~~~~~~~
        assert not full_dataset.is_file(), "shouldnt be"
        processed_path = str(full_dataset)

        # read data into huge `Data` list.
        data_list = []

        ds_base = Path(self.root)
        print(f"Creating {self.split} dataset at {str(ds_base)}")
        # TODO change this line to go to the new format
        # out_base = ds_base / ('ir_' + self.split + '_programl')
        # assert out_base.exists(), f"{out_base} doesn't exist!"
        # TODO collect .ll.pickle instead and call nx2data on the fly!
        print(f"=== DATASET {str(ds_base)}: Collecting .data.p files into dataset")

        # files = list(ds_base.rglob('*.data.p'))
        # files = list(ds_base.rglob('*.ll.pickle'))
        files = list(ds_base.rglob("*.ll.p"))

        for file in tqdm.tqdm(files):
            if not file.is_file():
                continue
            try:
                nx_graph = load(file)
            except EOFError:
                print(f"Failing to unpickle bc. EOFError on {file}! Skipping ...")
                continue
            try:
                data = nx2data(nx_graph, ignore_profile_info=False)
                data_list.append(data)
            except IndexError:
                print(
                    f"Failing nx2data bc IndexError (prob. empty graph) on {file}! Skipping ..."
                )
                continue

        print(f" * COMPLETED * === DATASET {ds_base}: now pre-filtering...")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        print(
            f" * COMPLETED * === DATASET {ds_base}: Completed filtering, now pre_transforming..."
        )

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        print(f" * COMPLETED * === DATASET {ds_base}: saving to disk...")
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), processed_path)

        # maybe save train_subset as well
        if not tuple(self.train_subset) == (0, 100) and self.split not in [
            "val",
            "test",
        ]:
            self._save_train_subset()


class NCCDataset(InMemoryDataset):
    @deprecated
    def __init__(
        self,
        root=REPO_ROOT / "deeplearning/ml4pl/poj104/ncc_data",
        split="train",
        transform=None,
        pre_transform=None,
        train_subset=[0, 100],
        train_subset_seed=0,
    ):
        """
        NCC dataset

        Args:
            train_subset: [start_percentile, stop_percentile)    default [0,100).
                            sample a random (but fixed) train set of data in slice by percent, with given seed.
            train_subset_seed: seed for the train_subset fixed random permutation.

        """
        self.split = split
        self.train_subset = train_subset
        self.train_subset_seed = train_subset_seed
        super().__init__(root, transform, pre_transform)

        assert split in [
            "train"
        ], "The NCC dataset only has a 'train' split. use train_subset=[0,x] and [x, 100] for training and testing."
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """A list of files that need to be found in the raw_dir in order to skip the download"""
        return []  # not implemented here

    @property
    def processed_file_names(self):
        """A list of files in the processed_dir which needs to be found in order to skip the processing."""
        base = f"{self.split}_data.pt"

        if tuple(self.train_subset) == (0, 100) or self.split in ["val", "test"]:
            return [base]
        else:
            assert self.split == "train"
            return [
                f"{self.split}_data_subset_{self.train_subset[0]}_{self.train_subset[1]}_seed_{self.train_subset_seed}.pt"
            ]

    def download(self):
        """Download raw data to `self.raw_dir`"""
        pass  # not implemented

    def _save_train_subset(self):
        """saves a train_subset of self to file.
        Percentile slice is taken according to self.train_subset
        with a fixed random permutation with self.train_subset_seed.
        """
        import numpy as np

        perm = np.random.RandomState(self.train_subset_seed).permutation(len(self))

        # take slice of perm according to self.train_subset
        start = np.math.floor(len(self) / 100 * self.train_subset[0])
        stop = np.math.floor(len(self) / 100 * self.train_subset[1])
        perm = perm[start:stop]
        print(f"Fixed permutation starts with: {perm[:min(30, len(perm))]}")

        dataset = self.__indexing__(perm)

        data, slices = dataset.data, dataset.slices
        torch.save((data, slices), self.processed_paths[0])
        return

    def filter_max_num_nodes(self, max_num_nodes):
        idx = []
        for i, d in enumerate(self):
            if d.num_nodes <= max_num_nodes:
                idx.append(i)
        dataset = self.__indexing__(idx)
        print(
            f"Filtering out graphs larger than {max_num_nodes} yields a dataset with {len(dataset)}/{len(self)} samples remaining."
        )
        return dataset

    def process(self):
        """Processes raw data and saves it into the `processed_dir`.
        New implementation:
            Here specifically it will collect all '*.ll.pickle' files recursively from subdirectories of `root`
            and process the loaded nx graphs to Data.
        Old implementation:
            Instead of looking for .ll.pickle (nx graphs), we directly look for '*.data.p' files.
        """
        # check if we need to create the full dataset:
        full_dataset = Path(self.processed_dir) / f"{self.split}_data.pt"
        if full_dataset.is_file():
            assert self.split == "train", "here shouldnt be reachable."
            print(
                f"Full dataset found. Generating train_subset={self.train_subset} with seed={self.train_subset_seed}"
            )
            # just get the split and save it
            self.data, self.slices = torch.load(full_dataset)
            self._save_train_subset()
            print(
                f"Saved train_subset={self.train_subset} with seed={self.train_subset_seed} to disk."
            )
            return

        # ~~~~~ we need to create the full dataset ~~~~~~~~~~~
        assert not full_dataset.is_file(), "shouldnt be"
        processed_path = str(full_dataset)

        # read data into huge `Data` list.
        data_list = []

        ds_base = Path(self.root)
        print(f"Creating {self.split} dataset at {str(ds_base)}")
        # TODO change this line to go to the new format
        # out_base = ds_base / ('ir_' + self.split + '_programl')
        # assert out_base.exists(), f"{out_base} doesn't exist!"
        # TODO collect .ll.pickle instead and call nx2data on the fly!
        print(f"=== DATASET {str(ds_base)}: Collecting .data.p files into dataset")

        # files = list(ds_base.rglob('*.data.p'))
        # files = list(ds_base.rglob('*.ll.pickle'))
        files = list(ds_base.rglob("*.ll.p"))

        for file in tqdm.tqdm(files):
            if not file.is_file():
                continue
            try:
                nx_graph = load(file)
            except EOFError:
                print(f"Failing to unpickle bc. EOFError on {file}! Skipping ...")
                continue
            try:
                data = nx2data(nx_graph)
                data_list.append(data)
            except IndexError:
                print(
                    f"Failing nx2data bc IndexError (prob. empty graph) on {file}! Skipping ..."
                )
                continue

        print(f" * COMPLETED * === DATASET {ds_base}: now pre-filtering...")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        print(
            f" * COMPLETED * === DATASET {ds_base}: Completed filtering, now pre_transforming..."
        )

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        print(f" * COMPLETED * === DATASET {ds_base}: saving to disk...")
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), processed_path)

        # maybe save train_subset as well
        if not tuple(self.train_subset) == (0, 100) and self.split not in [
            "val",
            "test",
        ]:
            self._save_train_subset()


class LegacyNCCDataset(InMemoryDataset):
    @deprecated
    def __init__(
        self,
        root="deeplearning/ml4pl/poj104/unsupervised_ncc_data",
        split="train",
        transform=None,
        pre_transform=None,
        train_subset=[0, 100],
        train_subset_seed=0,
    ):
        """
        Args:
            train_subset: [start_percentile, stop_percentile)    default [0,100).
                            sample a random (but fixed) train set of data in slice by percent, with given seed.
            train_subset_seed: seed for the train_subset fixed random permutation.

        """
        self.split = split
        self.train_subset = train_subset
        self.train_subset_seed = train_subset_seed
        super().__init__(root, transform, pre_transform)

        assert split in [
            "train"
        ], "The NCC dataset only has a 'train' split. use train_subset=[0,x] and [x, 100] for training and testing."
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """A list of files that need to be found in the raw_dir in order to skip the download"""
        return []  # not implemented here

    @property
    def processed_file_names(self):
        """A list of files in the processed_dir which needs to be found in order to skip the processing."""
        base = f"{self.split}_data.pt"

        if tuple(self.train_subset) == (0, 100) or self.split in ["val", "test"]:
            return [base]
        else:
            assert self.split == "train"
            return [
                f"{self.split}_data_subset_{self.train_subset[0]}_{self.train_subset[1]}_seed_{self.train_subset_seed}.pt"
            ]

    def download(self):
        """Download raw data to `self.raw_dir`"""
        pass  # not implemented

    def _save_train_subset(self):
        """saves a train_subset of self to file.
        Percentile slice is taken according to self.train_subset
        with a fixed random permutation with self.train_subset_seed.
        """
        import numpy as np

        perm = np.random.RandomState(self.train_subset_seed).permutation(len(self))

        # take slice of perm according to self.train_subset
        start = np.math.floor(len(self) / 100 * self.train_subset[0])
        stop = np.math.floor(len(self) / 100 * self.train_subset[1])
        perm = perm[start:stop]
        print(f"Fixed permutation starts with: {perm[:min(30, len(perm))]}")

        dataset = self.__indexing__(perm)

        data, slices = dataset.data, dataset.slices
        torch.save((data, slices), self.processed_paths[0])
        return

    def filter_max_num_nodes(self, max_num_nodes):
        idx = []
        for i, d in enumerate(self):
            if d.num_nodes <= max_num_nodes:
                idx.append(i)
        dataset = self.__indexing__(idx)
        print(
            f"Filtering out graphs larger than {max_num_nodes} yields a dataset with {len(dataset)}/{len(self)} samples remaining."
        )
        return dataset

    def process(self):
        """Processes raw data and saves it into the `processed_dir`.
        New implementation:
            Here specifically it will collect all '*.ll.pickle' files recursively from subdirectories of `root`
            and process the loaded nx graphs to Data.
        Old implementation:
            Instead of looking for .ll.pickle (nx graphs), we directly look for '*.data.p' files.
        """
        # check if we need to create the full dataset:
        full_dataset = Path(self.processed_dir) / f"{self.split}_data.pt"
        if full_dataset.is_file():
            assert self.split == "train", "here shouldnt be reachable."
            print(
                f"Full dataset found. Generating train_subset={self.train_subset} with seed={self.train_subset_seed}"
            )
            # just get the split and save it
            self.data, self.slices = torch.load(full_dataset)
            self._save_train_subset()
            print(
                f"Saved train_subset={self.train_subset} with seed={self.train_subset_seed} to disk."
            )
            return

        # ~~~~~ we need to create the full dataset ~~~~~~~~~~~
        assert not full_dataset.is_file(), "shouldnt be"
        processed_path = str(full_dataset)

        # read data into huge `Data` list.
        data_list = []

        ds_base = Path(self.root)
        print(f"Creating {self.split} dataset at {str(ds_base)}")
        # TODO change this line to go to the new format
        # out_base = ds_base / ('ir_' + self.split + '_programl')
        # assert out_base.exists(), f"{out_base} doesn't exist!"
        # TODO collect .ll.pickle instead and call nx2data on the fly!
        print(f"=== DATASET {str(ds_base)}: Collecting .data.p files into dataset")

        files = list(ds_base.rglob("*.data.p"))
        for file in tqdm.tqdm(files):
            if not file.is_file():
                continue
            data = load(file)
            data_list.append(data)

        print(f" * COMPLETED * === DATASET {ds_base}: now pre-filtering...")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        print(
            f" * COMPLETED * === DATASET {ds_base}: Completed filtering, now pre_transforming..."
        )

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        print(f" * COMPLETED * === DATASET {ds_base}: saving to disk...")
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), processed_path)

        # maybe save train_subset as well
        if not tuple(self.train_subset) == (0, 100) and self.split not in [
            "val",
            "test",
        ]:
            self._save_train_subset()


class ThreadcoarseningDataset(InMemoryDataset):
    @deprecated
    def __init__(
        self,
        root="deeplearning/ml4pl/poj104/threadcoarsening_data",
        split="fail_fast",
        transform=None,
        pre_transform=None,
        train_subset=[0, 100],
        train_subset_seed=0,
    ):
        """
        Args:
            train_subset: [start_percentile, stop_percentile)    default [0,100).
                            sample a random (but fixed) train set of data in slice by percent, with given seed.
            train_subset_seed: seed for the train_subset fixed random permutation.
            split: 'amd' or 'nvidia'

        """
        assert split in [
            "Cypress",
            "Tahiti",
            "Fermi",
            "Kepler",
        ], f"Split is {split}, but has to be 'Cypress', 'Tahiti', 'Fermi', or  'Kepler'"
        self.split = split
        self.train_subset = train_subset
        self.train_subset_seed = train_subset_seed
        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "threadcoarsening_data.zip"

    @property
    def processed_file_names(self):
        base = f"{self.split}_data.pt"

        if tuple(self.train_subset) == (0, 100):
            return [base]
        else:
            return [
                f"{self.split}_data_subset_{self.train_subset[0]}_{self.train_subset[1]}_seed_{self.train_subset_seed}.pt"
            ]

    def download(self):
        # download to self.raw_dir
        pass

    def return_cross_validation_splits(self, split):
        assert self.train_subset == [
            0,
            100,
        ], "Do cross-validation on the whole dataset!"
        assert (
            split <= 16 and split >= 0
        ), f"This dataset shall be 17-fold (leave one out) cross-validated, but split={split}."
        # leave one out
        n_splits = 17
        train_idx = list(range(n_splits))
        train_idx.remove(split)
        train_data = self.__indexing__(train_idx)
        test_data = self.__indexing__([split])
        return train_data, test_data

    def _save_train_subset(self):
        """saves a train_subset of self to file.
        Percentile slice is taken according to self.train_subset
        with a fixed random permutation with self.train_subset_seed.
        """
        import numpy as np

        perm = np.random.RandomState(self.train_subset_seed).permutation(len(self))

        # take slice of perm according to self.train_subset
        start = np.math.floor(len(self) / 100 * self.train_subset[0])
        stop = np.math.floor(len(self) / 100 * self.train_subset[1])
        perm = perm[start:stop]
        print(f"Fixed permutation starts with: {perm[:min(100, len(perm))]}")

        dataset = self.__indexing__(perm)

        data, slices = dataset.data, dataset.slices
        torch.save((data, slices), self.processed_paths[0])
        return

    def platform2str(self, platform):
        if platform == "Fermi":
            return "NVIDIA GTX 480"
        elif platform == "Kepler":
            return "NVIDIA Tesla K20c"
        elif platform == "Cypress":
            return "AMD Radeon HD 5900"
        elif platform == "Tahiti":
            return "AMD Tahiti 7970"
        else:
            raise LookupError

    def _get_all_runtimes(self, platform, df, oracles):
        all_runtimes = {}
        for kernel in oracles["kernel"]:
            kernel_r = []
            for cf in [1, 2, 4, 8, 16, 32]:
                row = df[(df["kernel"] == kernel) & (df["cf"] == cf)]
                if len(row) == 1:
                    value = float(row[f"runtime_{platform}"].values)
                    if math.isnan(value):
                        print(
                            f"WARNING: Dataset contain NaN value (missing entry in runtimes most likely). kernel={kernel}, cf={cf}, value={row}.Replacing by infinity!."
                        )
                        value = float("inf")
                    kernel_r.append(value)
                elif len(row) == 0:
                    print(
                        f" kernel={kernel:>20} is missing cf={cf}. Ad-hoc inserting result from last existing coarsening factor."
                    )
                    kernel_r.append(kernel_r[-1])
                else:
                    raise
            all_runtimes[kernel] = kernel_r
        return all_runtimes

    def process(self):
        # check if we need to create the full dataset:
        full_dataset = Path(self.processed_dir) / f"{self.split}_data.pt"
        if full_dataset.is_file():
            print(
                f"Full dataset {full_dataset.name} found. Generating train_subset={self.train_subset} with seed={self.train_subset_seed}"
            )
            # just get the split and save it
            self.data, self.slices = torch.load(full_dataset)
            self._save_train_subset()
            print(
                f"Saved train_subset={self.train_subset} with seed={self.train_subset_seed} to disk."
            )
            return

        # ~~~~~ we need to create the full dataset ~~~~~~~~~~~
        assert not full_dataset.is_file(), "shouldnt be"
        processed_path = str(full_dataset)

        root = Path(self.root)
        # Load runtime data
        oracle_file = root / "pact-2014-oracles.csv"
        oracles = pd.read_csv(oracle_file)

        runtimes_file = root / "pact-2014-runtimes.csv"
        df = pd.read_csv(runtimes_file)

        print("\tReading data from", oracle_file, "\n\tand", runtimes_file)

        # get all runtime info per kernel
        runtimes = self._get_all_runtimes(self.split, df=df, oracles=oracles)

        # get oracle labels
        cfs = [1, 2, 4, 8, 16, 32]
        y = np.array(
            [cfs.index(int(x)) for x in oracles["cf_" + self.split]], dtype=np.int64
        )

        # sanity check oracles against min runtimes
        for i, (k, v) in enumerate(runtimes.items()):
            assert int(y[i]) == np.argmin(
                v
            ), f"{i}: {k} {v}, argmin(v): {np.argmin(v)} vs. oracles data {int(y[i])}."

        # Add attributes to graphs
        data_list = []

        kernels = oracles["kernel"].values  # list of strings of kernel names

        for kernel in kernels:
            # legacy
            # file = root / 'kernels_ir_programl' / (kernel + '.data.p')
            file = root / "kernels_ir" / (kernel + ".ll.p")
            assert file.exists(), f"input file not found: {file}"
            # with open(file, 'rb') as f:
            #    data = pickle.load(f)
            g = load(file)
            data = nx2data(g)
            # add attributes
            data["y"] = torch.tensor([np.argmin(runtimes[kernel])], dtype=torch.long)
            data["runtimes"] = torch.tensor([runtimes[kernel]])
            data_list.append(data)

        ##################################

        print(
            f" * COMPLETED * === DATASET Threadcoarsening-{self.split}: now pre-filtering..."
        )

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        print(
            f" * COMPLETED * === DATASET Threadcoarsening-{self.split}: Completed filtering, now pre_transforming..."
        )

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), processed_path)

        # maybe save train_subset as well
        if not tuple(self.train_subset) == (0, 100) and self.split not in [
            "val",
            "test",
        ]:
            self._save_train_subset()


class DevmapDataset(InMemoryDataset):
    def __init__(
        self,
        root="deeplearning/ml4pl/poj104/devmap_data",
        split="fail",
        transform=None,
        pre_transform=None,
        train_subset=[0, 100],
        train_subset_seed=0,
        cdfg: bool = False,
        ablation_vocab: AblationVocab = AblationVocab.NONE,
    ):
        """
        Args:
            train_subset: [start_percentile, stop_percentile)    default [0,100).
                            sample a random (but fixed) train set of data in slice by percent, with given seed.
            train_subset_seed: seed for the train_subset fixed random permutation.
            split: 'amd' or 'nvidia'
            cdfg: Use CDFG graph representation.
        """
        assert split in [
            "amd",
            "nvidia",
        ], f"Split is {split}, but has to be 'amd' or 'nvidia'"
        self.split = split
        self.train_subset = train_subset
        self.train_subset_seed = train_subset_seed
        self.cdfg = cdfg
        self.ablation_vocab = ablation_vocab
        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "devmap_data.zip"

    @property
    def processed_file_names(self):
        base = filename(self.split, self.cdfg, self.ablation_vocab)

        if tuple(self.train_subset) == (0, 100):
            return [base]
        else:
            return [
                f"{name}_data_subset_{self.train_subset[0]}_{self.train_subset[1]}_seed_{self.train_subset_seed}.pt"
            ]

    def download(self):
        # download to self.raw_dir
        pass

    def return_cross_validation_splits(self, split):
        assert self.train_subset == [
            0,
            100,
        ], "Do cross-validation on the whole dataset!"
        # num_samples = len(self)
        # perm = np.random.RandomState(self.train_subset_seed).permutation(len(self))

        # 10-fold cross-validation
        n_splits = 10
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        (train_index, test_index) = list(kf.split(self.data.y, self.data.y))[split]
        train_data = self.__indexing__(train_index)
        test_data = self.__indexing__(test_index)
        return train_data, test_data

    def _save_train_subset(self):
        """saves a train_subset of self to file.
        Percentile slice is taken according to self.train_subset
        with a fixed random permutation with self.train_subset_seed.
        """
        perm = np.random.RandomState(self.train_subset_seed).permutation(len(self))

        # take slice of perm according to self.train_subset
        start = np.math.floor(len(self) / 100 * self.train_subset[0])
        stop = np.math.floor(len(self) / 100 * self.train_subset[1])
        perm = perm[start:stop]
        print(f"Fixed permutation starts with: {perm[:min(100, len(perm))]}")

        dataset = self.__indexing__(perm)

        data, slices = dataset.data, dataset.slices
        torch.save((data, slices), self.processed_paths[0])
        return

    def process(self):
        # check if we need to create the full dataset:
        name = filename(self.split, self.cdfg, self.ablation_vocab)
        full_dataset = Path(self.processed_dir) / name
        if full_dataset.is_file():
            print(
                f"Full dataset {full_dataset.name} found. Generating train_subset={self.train_subset} with seed={self.train_subset_seed}"
            )
            # just get the split and save it
            self.data, self.slices = torch.load(full_dataset)
            self._save_train_subset()
            print(
                f"Saved train_subset={self.train_subset} with seed={self.train_subset_seed} to disk."
            )
            return

        # ~~~~~ we need to create the full dataset ~~~~~~~~~~~
        assert not full_dataset.is_file(), "shouldnt be"
        processed_path = str(full_dataset)

        vocab = load_vocabulary(CDFG_VOCABULARY if self.cdfg else PROGRAML_VOCABULARY)
        assert len(vocab) > 0, "vocab is empty :|"

        root = Path(self.root)

        # Get list of source file names and attributes
        input_files = list((root / f"graphs_{self.split}").iterdir())

        num_files = len(input_files)
        print("\n--- Preparing to read", num_files, "input files")

        # read data into huge `Data` list.

        data_list = []
        for i in tqdm.tqdm(range(num_files)):
            filename = input_files[i]

            proto = load(filename, cdfg=self.cdfg)
            data = nx2data(proto, vocabulary=vocab, ablate_vocab=self.ablation_vocab)

            # Add the features and label.
            proto_features = proto.features.feature
            data["y"] = torch.tensor(
                proto_features["devmap_label"].int64_list.value[0]
            ).view(1)
            data["aux_in"] = torch.tensor(
                [
                    proto_features["transfer_bytes"].int64_list.value[0],
                    proto_features["wgsize"].int64_list.value[0],
                ]
            )

            data_list.append(data)

        ##################################

        print(f" * COMPLETED * === DATASET Devmap-{name}: now pre-filtering...")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        print(
            f" * COMPLETED * === DATASET Devmap-{name}: Completed filtering, now pre_transforming..."
        )

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), processed_path)

        # maybe save train_subset as well
        if not tuple(self.train_subset) == (0, 100):
            self._save_train_subset()


class POJ104Dataset(InMemoryDataset):
    def __init__(
        self,
        root="deeplearning/ml4pl/poj104/classifyapp_data",
        split="fail",
        transform=None,
        pre_transform=None,
        train_subset=[0, 100],
        train_subset_seed=0,
        cdfg: bool = False,
        ablation_vocab: AblationVocab = AblationVocab.NONE,
    ):
        """
        Args:
            train_subset: [start_percentile, stop_percentile)    default [0,100).
                            sample a random (but fixed) train set of data in slice by percent, with given seed.
            train_subset_seed: seed for the train_subset fixed random permutation.
            cdfg: Use the CDFG graph format and vocabulary.
        """
        self.split = split
        self.train_subset = train_subset
        self.train_subset_seed = train_subset_seed
        self.cdfg = cdfg
        self.ablation_vocab = ablation_vocab
        super().__init__(root, transform, pre_transform)

        assert split in ["train", "val", "test"]
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "classifyapp_data.zip"  # ['ir_val', 'ir_val_programl']

    @property
    def processed_file_names(self):
        base = filename(self.split, self.cdfg, self.ablation_vocab)

        if tuple(self.train_subset) == (0, 100) or self.split in ["val", "test"]:
            return [base]
        else:
            assert self.split == "train"
            return [
                f"{self.split}_data_subset_{self.train_subset[0]}_{self.train_subset[1]}_seed_{self.train_subset_seed}.pt"
            ]

    def download(self):
        # download to self.raw_dir
        pass

    def _save_train_subset(self):
        """saves a train_subset of self to file.
        Percentile slice is taken according to self.train_subset
        with a fixed random permutation with self.train_subset_seed.
        """
        import numpy as np

        perm = np.random.RandomState(self.train_subset_seed).permutation(len(self))

        # take slice of perm according to self.train_subset
        start = np.math.floor(len(self) / 100 * self.train_subset[0])
        stop = np.math.floor(len(self) / 100 * self.train_subset[1])
        perm = perm[start:stop]
        print(f"Fixed permutation starts with: {perm[:min(100, len(perm))]}")

        dataset = self.__indexing__(perm)

        data, slices = dataset.data, dataset.slices
        torch.save((data, slices), self.processed_paths[0])
        return

    def process(self):
        # hardcoded
        num_classes = 104

        # check if we need to create the full dataset:
        full_dataset = Path(self.processed_dir) / filename(
            self.split, self.cdfg, self.ablation_vocab
        )
        if full_dataset.is_file():
            assert self.split == "train", "here shouldnt be reachable."
            print(
                f"Full dataset found. Generating train_subset={self.train_subset} with seed={self.train_subset_seed}"
            )
            # just get the split and save it
            self.data, self.slices = torch.load(full_dataset)
            self._save_train_subset()
            print(
                f"Saved train_subset={self.train_subset} with seed={self.train_subset_seed} to disk."
            )
            return

        # ~~~~~ we need to create the full dataset ~~~~~~~~~~~
        assert not full_dataset.is_file(), "shouldnt be"
        processed_path = str(full_dataset)

        # get vocab first
        vocab = load_vocabulary(CDFG_VOCABULARY if self.cdfg else PROGRAML_VOCABULARY)
        assert len(vocab) > 0, "vocab is empty :|"
        # read data into huge `Data` list.
        data_list = []

        ds_base = Path(self.root)
        print(f"Creating {self.split} dataset at {str(ds_base)}")

        split_folder = ds_base / (self.split)
        assert split_folder.exists(), f"{split_folder} doesn't exist!"

        # collect .pb and call nx2data on the fly!
        print(
            f"=== DATASET {split_folder}: Collecting ProgramGraph.pb files into dataset"
        )

        # only take files from subfolders (with class names!) recursively
        files = [x for x in split_folder.rglob("*ProgramGraph.pb")]
        assert len(files) > 0, "no files collected. error."
        for file in tqdm.tqdm(files):
            # skip classes that are larger than what config says to enable debugging with less data
            # class_label = int(file.parent.name) - 1  # let classes start from 0.
            # if class_label >= num_classes:
            #    continue

            graph = load(file, cdfg=self.cdfg)
            data = nx2data(
                graph=graph,
                vocabulary=vocab,
                ablate_vocab=self.ablation_vocab,
                y_feature_name="poj104_label",
            )
            data_list.append(data)

        print(f" * COMPLETED * === DATASET {split_folder}: now pre-filtering...")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        print(
            f" * COMPLETED * === DATASET {split_folder}: Completed filtering, now pre_transforming..."
        )

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), processed_path)

        # maybe save train_subset as well
        if not tuple(self.train_subset) == (0, 100) and self.split not in [
            "val",
            "test",
        ]:
            self._save_train_subset()


class LegacyPOJ104Dataset(InMemoryDataset):
    @deprecated
    def __init__(
        self,
        root="deeplearning/ml4pl/poj104/classifyapp_data",
        split="fail",
        transform=None,
        pre_transform=None,
        train_subset=[0, 100],
        train_subset_seed=0,
    ):
        """
        Args:
            train_subset: [start_percentile, stop_percentile)    default [0,100).
                            sample a random (but fixed) train set of data in slice by percent, with given seed.
            train_subset_seed: seed for the train_subset fixed random permutation.

        """
        self.split = split
        self.train_subset = train_subset
        self.train_subset_seed = train_subset_seed
        super().__init__(root, transform, pre_transform)

        assert split in ["train", "val", "test"]
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "classifyapp_data.zip"  # ['ir_val', 'ir_val_programl']

    @property
    def processed_file_names(self):
        base = f"{self.split}_data.pt"

        if tuple(self.train_subset) == (0, 100) or self.split in ["val", "test"]:
            return [base]
        else:
            assert self.split == "train"
            return [
                f"{self.split}_data_subset_{self.train_subset[0]}_{self.train_subset[1]}_seed_{self.train_subset_seed}.pt"
            ]

    def download(self):
        # download to self.raw_dir
        pass

    def _save_train_subset(self):
        """saves a train_subset of self to file.
        Percentile slice is taken according to self.train_subset
        with a fixed random permutation with self.train_subset_seed.
        """
        import numpy as np

        perm = np.random.RandomState(self.train_subset_seed).permutation(len(self))

        # take slice of perm according to self.train_subset
        start = np.math.floor(len(self) / 100 * self.train_subset[0])
        stop = np.math.floor(len(self) / 100 * self.train_subset[1])
        perm = perm[start:stop]
        print(f"Fixed permutation starts with: {perm[:min(100, len(perm))]}")

        dataset = self.__indexing__(perm)

        data, slices = dataset.data, dataset.slices
        torch.save((data, slices), self.processed_paths[0])
        return

    def process(self):
        # hardcoded
        num_classes = 104

        # check if we need to create the full dataset:
        full_dataset = Path(self.processed_dir) / f"{self.split}_data.pt"
        if full_dataset.is_file():
            assert self.split == "train", "here shouldnt be reachable."
            print(
                f"Full dataset found. Generating train_subset={self.train_subset} with seed={self.train_subset_seed}"
            )
            # just get the split and save it
            self.data, self.slices = torch.load(full_dataset)
            self._save_train_subset()
            print(
                f"Saved train_subset={self.train_subset} with seed={self.train_subset_seed} to disk."
            )
            return

        # ~~~~~ we need to create the full dataset ~~~~~~~~~~~
        assert not full_dataset.is_file(), "shouldnt be"
        processed_path = str(full_dataset)

        # read data into huge `Data` list.
        data_list = []

        ds_base = Path(self.root)
        print(f"Creating {self.split} dataset at {str(ds_base)}")
        # TODO change this line to go to the new format
        out_base = ds_base / ("ir_" + self.split + "_programl")
        assert out_base.exists(), f"{out_base} doesn't exist!"
        # TODO collect .ll.pickle instead and call nx2data on the fly!
        print(f"=== DATASET {out_base}: Collecting .data.p files into dataset")

        folders = [
            x
            for x in out_base.glob("*")
            if x.is_dir() and x.name not in ["_nx", "_tuples"]
        ]
        for folder in tqdm.tqdm(folders):
            # skip classes that are larger than what config says to enable debugging with less data
            if int(folder.name) > num_classes:
                continue
            for k, file in enumerate(folder.glob("*.data.p")):
                with open(file, "rb") as f:
                    data = pickle.load(f)
                data_list.append(data)

        print(f" * COMPLETED * === DATASET {out_base}: now pre-filtering...")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        print(
            f" * COMPLETED * === DATASET {out_base}: Completed filtering, now pre_transforming..."
        )

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), processed_path)

        # maybe save train_subset as well
        if not tuple(self.train_subset) == (0, 100) and self.split not in [
            "val",
            "test",
        ]:
            self._save_train_subset()

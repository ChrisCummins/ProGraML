import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, Batch
from torch._six import container_abcs, string_classes, int_classes


class DataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[], **kwargs):
        def collate(batch):
            elem = batch[0]
            if isinstance(elem, Data):
                return Batch.from_data_list(batch, follow_batch)
            elif isinstance(elem, float):
                return torch.tensor(batch, dtype=torch.float)
            elif isinstance(elem, int_classes):
                return torch.tensor(batch)
            elif isinstance(elem, string_classes):
                return batch
            elif isinstance(elem, container_abcs.Mapping):
                return {key: collate([d[key] for d in batch]) for key in elem}
            elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
                return type(elem)(*(collate(s) for s in zip(*batch)))
            elif isinstance(elem, container_abcs.Sequence):
                return [collate(s) for s in zip(*batch)]
            raise TypeError(
                "DataLoader found invalid type: {}".format(type(elem).__name__)
            )

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda batch: collate(batch),
            **kwargs,
        )


class NodeLimitedDataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        follow_batch=[],
        max_num_nodes=None,
        warn_on_limit=False,
        **kwargs,
    ):
        self.max_num_nodes = max_num_nodes

        def collate(batch):
            elem = batch[0]
            if isinstance(elem, Data):
                # greedily add all samples that fit within self.max_num_nodes
                # and silently discard all others
                if max_num_nodes is not None:
                    num_nodes = 0
                    limited_batch = []
                    for elem in batch:
                        if num_nodes + elem.num_nodes <= self.max_num_nodes:
                            limited_batch.append(elem)
                            num_nodes += elem.num_nodes
                        else:  # for debugging
                            pass
                    if len(limited_batch) < len(batch):
                        if warn_on_limit:
                            print(
                                f"dropped {len(batch) - len(limited_batch)} graphs from batch!"
                            )
                    assert (
                        limited_batch != []
                    ), f"limited batch is empty! original batch was {batch}"
                    return Batch.from_data_list(limited_batch, follow_batch)
                else:
                    return Batch.from_data_list(batch, follow_batch)
            elif isinstance(elem, float):
                return torch.tensor(batch, dtype=torch.float)
            elif isinstance(elem, int_classes):
                return torch.tensor(batch)
            elif isinstance(elem, string_classes):
                return batch
            elif isinstance(elem, container_abcs.Mapping):
                return {key: collate([d[key] for d in batch]) for key in elem}
            elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
                return type(elem)(*(collate(s) for s in zip(*batch)))
            elif isinstance(elem, container_abcs.Sequence):
                return [collate(s) for s in zip(*batch)]

            raise TypeError("DataLoader found invalid type: {}".format(type(elem)))

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda batch: collate(batch),
            **kwargs,
        )

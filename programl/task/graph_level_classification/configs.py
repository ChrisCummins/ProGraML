# Copyright 2019 the ProGraML authors.
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
"""Configs"""
from programl.task.graph_level_classification.dataset import AblationVocab
from programl.util.py.deprecated import deprecated


class ProGraMLBaseConfig(object):
    def __init__(self):
        self.name = self.__class__.__name__

        # Training Hyperparameters
        self.num_epochs = 25
        self.batch_size = 128
        # limit the number of nodes per batch to a sensible maximum
        # by possibly discarding certain samples from the batch.
        self.max_num_nodes = 200000
        self.lr: float = 0.00025
        self.patience = 10000
        self.clip_grad_norm: float = 0.0
        self.train_subset = [0, 100]
        self.random_seed: int = 42

        # Readout
        self.output_dropout: float = 0.0

        # Model Hyperparameters
        self.emb_size: int = 200
        self.edge_type_count: int = 3

        self.vocab_size: int = 8568
        self.cdfg_vocab: bool = False

        # ABLATION OPTIONS
        # NONE = 0 No ablation - use the full vocabulary (default).
        # NO_VOCAB = 1 Ignore the vocabulary - every node has an x value of 0.
        # NODE_TYPE_ONLY = 2 Use a 3-element vocabulary based on the node type:
        #    0 - Instruction node
        #    1 - Variable node
        #    2 - Constant node
        self.ablation_vocab: AblationVocab = 0  # 0 NONE, 1 NO_VOCAB, 2 NODE_TYPE_ONLY

        # inst2vec_embeddings can now be 'none' as well!
        # this reduces the tokens that the network sees to only
        # !IDENTIFIERs and !UNK statements
        #  One of {zero, constant, random, random_const, finetune, none}
        self.inst2vec_embeddings = "random"

        self.ablate_structure = None  # one of {control,data,call}

    @classmethod
    def from_dict(cls, params):
        """instantiate Config from params dict that overrides default values where given."""
        config = cls()
        if params is None:
            return config

        for key in params:
            if hasattr(config, key):
                setattr(config, key, params[key])
            else:
                print(
                    f"(*CONFIG FROM DICT*  Default {config.name} doesn't have a key {key}. Will add key to config anyway!"
                )
                setattr(config, key, params[key])
        return config

    def to_dict(self):
        config_dict = {
            a: getattr(self, a)
            for a in dir(self)
            if not a.startswith("__") and not callable(getattr(self, a))
        }
        return config_dict

    def check_equal(self, other):
        # take either config object or config_dict
        other_dict = other if isinstance(other, dict) else other.to_dict()
        if not self.to_dict() == other_dict:
            print(
                f"WARNING: GGNNConfig.check_equal() FAILED:\nself and other are unequal: "
                f"The difference is {set(self.to_dict()) ^ set(other.to_dict())}.\n self={self.to_dict()}\n other={other_dict}"
            )


class GGNN_POJ104_Config(ProGraMLBaseConfig):
    def __init__(self):
        super().__init__()
        ###############
        # Model Hyperparameters
        self.gnn_layers: int = 8
        self.message_weight_sharing: int = 2
        self.update_weight_sharing: int = 2
        # self.message_timesteps: List[int] = [2, 2, 2, 2]
        # self.update_timesteps: List[int] = [2, 2, 2, 2]

        # currently only admits node types 0 and 1 for statements and identifiers.
        self.use_node_types = True
        self.use_edge_bias: bool = True
        self.position_embeddings: bool = True

        # Aggregate by mean or by sum
        self.msg_mean_aggregation: bool = True
        self.backward_edges: bool = True

        ###############
        # Regularization
        self.edge_weight_dropout: float = 0.0
        self.graph_state_dropout: float = 0.2

        ###############
        # Dataset inherent, don't change!
        self.num_classes: int = 104
        self.has_graph_labels: bool = True
        self.has_aux_input: bool = False

        # self.use_selector_embeddings: bool = False
        # self.selector_size: int = 2 if getattr(self, 'use_selector_embeddings', False) else 0
        # TODO(Zach) Maybe refactor non-rectangular edge passing matrices for independent hidden size.
        # hidden size of the whole model
        self.hidden_size: int = self.emb_size + getattr(self, "selector_size", 0)


class GGNN_Devmap_Config(GGNN_POJ104_Config):
    def __init__(self):
        super().__init__()
        # change default
        self.batch_size = 64
        self.lr = 2.5e-4
        self.num_epochs = 150
        self.graph_state_dropout = 0.0

        # Auxiliary Readout
        self.aux_use_better = False
        self.intermediate_loss_weight = 0.2
        self.aux_in_size = 2
        self.aux_in_layer_size = 32
        self.aux_in_log1p = True

        # Dataset inherent, don't change!
        self.num_classes: int = 2
        self.has_graph_labels: bool = True
        self.has_aux_input: bool = True


class GGNN_Threadcoarsening_Config(GGNN_POJ104_Config):
    @deprecated
    def __init__(self):
        super().__init__()
        # Dataset inherent, don't change!
        self.num_classes: int = 6
        self.has_graph_labels: bool = True
        # self.has_aux_input: bool = False


class GGNN_ForPretraining_Config(GGNN_POJ104_Config):
    @deprecated
    def __init__(self):
        super().__init__()
        # Pretraining Parameters
        self.mlm_probability = 0.15
        self.mlm_statements_only = True
        self.mlm_exclude_unk_tokens = True
        self.mlm_mask_token_id = 8568
        self.unk_token_id = 8564

        # set for pretraining to vocab_size + 1 [MASK]
        self.vocab_size = self.vocab_size + 1
        self.num_classes = self.vocab_size
        self.has_graph_labels: bool = False


class GraphTransformer_POJ104_Config(ProGraMLBaseConfig):
    @deprecated
    def __init__(self):
        super().__init__()
        ###### borrowed for debugging ##########

        # GGNNMessage Layer
        # self.msg_mean_aggregation: bool = True
        # self.use_edge_bias: bool = True

        ###############
        self.backward_edges: bool = True
        self.gnn_layers: int = 8
        self.message_weight_sharing: int = 2
        self.update_weight_sharing: int = 2
        # self.layer_timesteps: List[int] = [1, 1, 1, 1, 1, 1, 1, 1] #[2, 2, 2, 2]
        self.use_node_types: bool = False

        # Dataset Specific, don't change!
        self.num_classes: int = 104
        self.has_graph_labels: bool = True
        self.hidden_size: int = self.emb_size + getattr(self, "selector_size", 0)

        # Message:
        self.position_embeddings: bool = True
        #  Self-Attn Layer
        self.attn_bias = True
        self.attn_num_heads = 5  # 8 # choose among 4,5,8,10 for emb_sz 200
        self.attn_dropout = 0.1
        self.attn_v_pos = False

        # Update:

        # Transformer Update Layer
        self.update_layer: str = "ff"  # or 'gru'
        self.tfmr_act = "gelu"  # relu or gelu, default relu
        self.tfmr_dropout = 0.2  # default 0.1
        self.tfmr_ff_sz = 512  # 512 # ~ 2.5 model_dim (Bert: 768 - 2048, Trfm: base 512 - 2048, big 1024 - 4096)

        # Optionally: GGNN Update Layer
        # self.update_layer: str = 'gru' # or 'ff'
        # self.edge_weight_dropout: float = 0.0
        # self.graph_state_dropout: float = 0.2


class GraphTransformer_Devmap_Config(GraphTransformer_POJ104_Config):
    @deprecated
    def __init__(self):
        super().__init__()
        # change default
        self.batch_size = 64
        self.lr = 2.5e-4
        self.num_epochs = 600
        # self.graph_state_dropout = 0.0 #GGNN only

        # self.output_dropout # <- applies to Readout func!

        # Auxiliary Readout
        self.aux_use_better = False
        self.intermediate_loss_weight = 0.2
        self.aux_in_size = 2
        self.aux_in_layer_size = 32
        self.aux_in_log1p = True

        # Dataset inherent, don't change!
        self.num_classes: int = 2
        self.has_graph_labels: bool = True
        self.has_aux_input: bool = True


class GraphTransformer_Threadcoarsening_Config(GraphTransformer_POJ104_Config):
    @deprecated
    def __init__(self):
        super().__init__()
        self.lr = 5e-5  # 2.5-4?
        self.num_epochs = 600
        # Dataset inherent, don't change!
        self.num_classes: int = 6
        self.has_graph_labels: bool = True
        # self.has_aux_input: bool = False


class GraphTransformer_ForPretraining_Config(GraphTransformer_POJ104_Config):
    @deprecated
    def __init__(self):
        super().__init__()
        self.num_of_splits = 2
        # Pretraining Parameters
        self.mlm_probability = 0.15
        self.mlm_statements_only = True
        self.mlm_exclude_unk_tokens = True
        self.mlm_mask_token_id = 8568
        self.unk_token_id = 8564

        # set for pretraining to vocab_size + 1 [MASK]
        self.vocab_size = self.vocab_size + 1
        self.num_classes = self.vocab_size
        self.has_graph_labels: bool = False


class GGNN_BranchPrediction_Config(GGNN_POJ104_Config):
    @deprecated
    def __init__(self):
        super().__init__()
        self.batch_size = 4
        # self.use_tanh_readout = False !
        self.num_classes = 1
        self.has_graph_labels = False


class GraphTransformer_BranchPrediction_Config(GraphTransformer_POJ104_Config):
    @deprecated
    def __init__(self):
        super().__init__()
        self.batch_size = 4
        # self.use_tanh_readout = False !
        self.num_classes = 1
        self.has_graph_labels = False

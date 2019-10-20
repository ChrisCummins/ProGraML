# Graph-based Machine Learning for Programming Languages

**[Travis CI build status](https://travis-ci.com/ChrisCummins/ml4pl)**

## Setup

### Requirements

* Ubuntu >= 16.04
* Python >= 3.6
* MySQL

### Installation

```sh
$ ./configure
# Answer yes/no questions.
```

Run unit tests:

```sh
$ bazel test //deeplearning/ml4pl/...
```

### Create a bytecode database

Download the 
[POJ-104 IR dataset](https://polybox.ethz.ch/index.php/s/JOBjrfmAjOeWCyl/download) 
from:

Create a bytecode database:

```sh
$ bazel run //deeplearning/ml4pl/bytecode/create:import_from_poj104 -- \
    --db='file:///path/to/db.mysql?ml4pl_bytecode?charset=utf8' \
    --dataset=/path/to/root/of/dataset
```

## Running Experiments

### Classifyapp

Create the dataset:

```sh
$ bazel run //deeplearning/ml4pl/graphs/labelled/classifyapp:make_classifyapp_dataset -- \
    --bytecode_db='file:///var/phd/db/cc1.mysql?ml4pl_bytecode?charset=utf8' \
    --graph_db='file:///var/phd/db/cc1.mysql?ml4pl_classifyapp_xfg_poj104?charset=utf8' \
    --database_exporter_batch_size=5000 \
    --alsologtostderr
```

Train and evaluate a model:

```sh
$ bazel run //deeplearning/ml4pl/models/ggnn:ggnn_graph_classifier -- \
    --graph_db='file:///var/phd/db/cc1.mysql?ml4pl_classifyapp_xfg_poj104?charset=utf8' \
    --working_dir='/var/phd/shared/ml4pl/models/classifyapp_xfg_poj104' \
    --max_instance_count=0 \
    --layer_timesteps=2,2,2 \
    --alsologtostderr
```

View logs

```sh
$ tensorboard --logdir /var/phd/shared/ml4pl/models/classifyapp_xfg_poj104/tensorboard
```

### Reachability

Create the dataset:

```sh
$ bazel run //deeplearning/ml4pl/graphs/labelled/reachability:make_reachability_dataset -- \
    --bytecode_db='file:///var/phd/db/cc1.mysql?ml4pl_bytecode?charset=utf8' \
    --graph_db='file:///var/phd/db/cc1.mysql?ml4pl_reachability_poj104?charset=utf8' \
    --bytecode_split_type='poj104' \
    --alsologtostderr
```

Train a model on <= 10 step graphs:

```sh
$ bazel run //deeplearning/ml4pl/models/ggnn:ggnn_node_classifier -- \
    --graph_db='file:///var/phd/db/cc1.mysql?ml4pl_reachability_cfg_poj104?charset=utf8' \
    --working_dir='/var/phd/shared/ml4pl/models/reachability_poj104/m10' \
    --max_instance_count=10000 \
    --max_steps=10 \
    --alsologtostderr
```

View logs:

```sh
$ tensorboard --logdir /var/phd/shared/ml4pl/models/reachability_poj104/m10/tensorboard
```

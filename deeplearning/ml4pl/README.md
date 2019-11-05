# Graph-based Machine Learning for Programming Languages

**[Travis CI build status](https://travis-ci.com/ChrisCummins/ml4pl)**

## Setup

### Requirements

* Ubuntu >= 16.04
* Python >= 3.6
* Bazel == 0.26.0
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

### Setup database

Create a file with the username, password, and hostname of your MySQL server in the form:

```sh
$ cat /tmp/mysql.txt
mysql://user:pass@hostname/
```

### Create a bytecode database

Download the 
[POJ-104 IR dataset](https://polybox.ethz.ch/index.php/s/JOBjrfmAjOeWCyl/download) 
from:

Create a database and import the bytecodes:

```sh
$ bazel run //deeplearning/ml4pl/bytecode/create:import_from_poj104 -- \
    --db='file:///tmp/mysql.txt?ml4pl_bytecode?charset=utf8' \
    --dataset=/path/to/root/of/dataset
```

## Running Experiments

### Classifyapp

Create the dataset:

```sh
$ bazel run //deeplearning/ml4pl/graphs/labelled/classifyapp:make_classifyapp_dataset -- \
    --bytecode_db='file:///tmp/mysql.txt?ml4pl_bytecode?charset=utf8' \
    --graph_db='file:///tmp/mysql.txt?ml4pl_classifyapp_xfg_poj104?charset=utf8' \
    --database_exporter_batch_size=5000 \
    --alsologtostderr
```

Train and evaluate a model:

```sh
$ bazel run //deeplearning/ml4pl/models/ggnn:ggnn_graph_classifier -- \
    --graph_db='file:///tmp/mysql.txt?ml4pl_classifyapp_xfg_poj104?charset=utf8' \
    --working_dir='/var/ml4pl/models/classifyapp_xfg_poj104' \
    --layer_timesteps=2,2,2 \
    --alsologtostderr
```

View logs

```sh
$ tensorboard --logdir /var/ml4pl/models/classifyapp_xfg_poj104/tensorboard
```

### Reachability

Create the dataset:

```sh
$ bazel run //deeplearning/ml4pl/graphs/labelled/reachability:make_reachability_dataset -- \
    --bytecode_db='file:///tmp/mysql.txt?ml4pl_bytecode?charset=utf8' \
    --graph_db='file:///tmp/mysql.txt?ml4pl_reachability_poj104?charset=utf8' \
    --bytecode_split_type='poj104' \
    --reachability_dataset_max_instances_per_graph=3 \
    --alsologtostderr
```

Train a model on graphs requiring <= 15 time steps:

```sh
$ bazel run //deeplearning/ml4pl/models/ggnn:ggnn -- \
    --graph_db='file:///tmp/mysql.txt?ml4pl_reachability_cfg_poj104?charset=utf8' \
    --working_dir='/var/ml4pl/models/reachability_poj104' \
    --max_instance_count=10000 \
    --layer_timesteps=15 \
    --limit_data_flow_max_steps_required_to_message_passing_steps \
    --alsologtostderr
```

View logs:

```sh
$ tensorboard --logdir /var/ml4pl/models/reachability_poj104/tensorboard
```

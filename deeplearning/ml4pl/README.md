# Graph-based Machine Learning for Programming Languages

<!-- license -->
<a href="https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg?color=brightgreen">
</a>
<!-- Better code -->
<a href="https://bettercodehub.com/results/ChrisCummins/ProGraML">
  <img src="https://bettercodehub.com/edge/badge/ChrisCummins/ProGraML?branch=master">
</a>
<!-- Travis CI -->
<a href="https://travis-ci.org/ChrisCummins/ProGraML">
  <img src="https://img.shields.io/travis/ChrisCummins/ProGraML/master.svg">
</a>
<!-- commit counter -->
<a href="https://github.com/ChrisCummins/ProGraML/graphs/commit-activity">
  <img src="https://img.shields.io/github/commit-activity/y/ChrisCummins/ProGraML.svg?color=yellow">
</a>
<!-- repo size -->
<a href="https://github.com/ChrisCummins/ProGraML">
  <img src="https://img.shields.io/github/repo-size/ChrisCummins/ProGraML.svg">
</a>

## Setup

### Installation

See [INSTALL.md](/INSTALL.md). Build and run the test suite using:

```sh
$ bazel test //deeplearning/ml4pl/...
```

#### Setup a database

Create a file with the username, password, and hostname of your MySQL server in the form:

```sh
$ cat ~/mysql.txt
mysql://user:pass@hostname/
```

Then export the path of this file with a `file://` prefix:

```sh
$ export DB="file://$HOME/mysql.txt"
```

## Usage

### Step 1: Gather Intermediate Representations

TODO(github.com/ChrisCummins/ProGraML/issues/7). This is currently being
refactored, and the existing databases have been migrated from an old schema.


### Step 2: Generate Program Graphs

Generate a program graph protocol buffer from an LLVM IR file using:

```
$ bazel run //deeplearning/ml4pl/graphs/llvm2graph -- /tmp/foo.ll
```

The above tool is also available as a docker image:

```
$ docker pull chriscummins/llvm2graph:latest
$ docker run -i chriscummins/llvm2graph < /tmp/foo.ll
```

TODO(github.com/ChrisCummins/ProGraML/issues/2): The scripts for generating
graph protos from databases of IRs are currently being refactored.


### Step 3: Create Labelled Graphs

#### Data Flow Analyses

Generate a database of graphs labelled with data flow analyses using:

```sh
$ bazel run //deeplearning/ml4pl/graphs/labelled:flaky_make_data_flow_analysis_dataset -- \
    --proto_db="$DB?programl_graph_protos" \
    --graph_db="$DB?programl_$ANALYSIS" \
    --analysis=$ANALYSIS
```

Where `$ANALYSIS` is one of:

 1. `reachability` Control flow reachability.
 2. `domtree` Dominator trees.
 3. `liveness` Live-out variables.
 4. `datadep` Data dependencies.
 5. `subexpressions` Global common subexpressions.

Split one of the datasets intro {train,val,test} data using:

```sh
$ bazel run //deeplearning/ml4pl/graphs/labelled/dataflow:split -- \
    --graph_db="$DB?programl_reachability" \
    --ir_db="$DB?programl_ir"
```

Then copy those splits to the other datasets using:

```sh
$ bazel run //deeplearning/ml4pl/graphs/labelled/dataflow:split -- \
    --graph_db="$DB?programl_reachability" \
    --copy_splits_to="$DB?programl_domtree"

$ bazel run //deeplearning/ml4pl/graphs/labelled/dataflow:split -- \
    --graph_db="$DB?programl_reachability" \
    --copy_splits_to="$DB?programl_liveness"

$ bazel run //deeplearning/ml4pl/graphs/labelled/dataflow:split -- \
    --graph_db="$DB?programl_reachability" \
    --copy_splits_to="$DB?programl_datadep"

$ bazel run //deeplearning/ml4pl/graphs/labelled/dataflow:split -- \
    --graph_db="$DB?programl_reachability" \
    --copy_splits_to="$DB?programl_subexpressions"
```

#### Heterogeneous Device Mapping

We provide a dataset of heterogeneous device mapping benchmarks for two devices:
`nvidia_gtx_960` and `amd_tahiti_7970`. Generate the labelled datasets using:

```sh
$ bazel run //deeplearning/ml4pl/graphs/labelled/devmap:make_devmap_dataset -- \
    --ir_db="$DB?programl_ir" \
    --proto_db="$DB?programl_graph_protos" \
    --graph_db="$DB?programl_devmap_amd" \
    --gpu='amd_tahiti_7970'

$ bazel run //deeplearning/ml4pl/graphs/labelled/devmap:make_devmap_dataset -- \
    --ir_db="$DB?programl_ir" \
    --proto_db="$DB?programl_graph_protos" \
    --graph_db="$DB?programl_devmap_nvidia" \
    --gpu='nvidia_gtx_960'
```

We use stratified 10-fold cross-validation to evaluate models on these datasets
due to their small size. Create random 10-fold splits of the datasets using:

```sh
$ bazel run //deeplearning/ml4pl/graphs/labelled/devmap:split -- \
    --graph_db="$DB?programl_devmap_amd" \
    --k=10

$ bazel run //deeplearning/ml4pl/graphs/labelled/devmap:split -- \
    --graph_db="$DB?programl_devmap_nvidia" \
    --k=10
```

### Step 4: Train Models

Each of the models accept a common set of configuration flags, which you can see
using `--helpfull`. Some of the relevant options are:

* `--epoch_count`: The number of epochs to run the train/val loop for.
* `--test_on`: Determine when to run the test set. Possible values: none (never
  run the test set), every (test at the end of every epoch), improvement (test
  only when validation accuracy improves), improvent_and_last (test when
  validation accuracy improves, and on the last epoch), or best (restore the
  model with the best validation accuracy after training and test that).
* `--stop_at`: Permit the train/val/test loop to terminate before `epoch_count`
  iterations have completed. Valid options are: `val_acc=<float>` (stop if
  validation accuracy reaches the given value in the range [0,1]),
  `elapsed=<int>` (stop if the given number of seconds have elapsed, excluding
  the final test epoch if `--test_on=best` is set), or `patience=<int>` (stop if
  <int> epochs have been performed without an improvement in validation
  accuracy. Multiple options can be combined in a comma-separated list, e.g.
  `--stop_at=val_acc=.9999,elapsed=21600,patience=10` meaning stop if validation
  accuracy meets 99.99% or if 6 hours have elapsed or if 10 epochs have been
  performed without an improvement in validation accuracy.
* `--keep_checkpoints=all`: One of `{all,last}`
* `--detailed_batch_types=val,test`: A list of epoch types to make detailed batch
logs for, where the epoch type is one of `{train,val,test}`. Detailed batch logs
store more information about the model behavior, but are not required for
computing aggregate model performance and can
grow large.
* `--keep_detailed_batches=all`: Determines whether to remove old detailed
batches over the lifetime of a model. Valid options are `all`: keep all detailed
batches, or `last_epoch`: delete old detailed batches at the end of every epoch.
* `--tag=<string>`: An arbitrary string which can be used for grouping the
results of multiple models when aggregating results.
* `--noreuse_tag`: When setting a `--tag` for a run, check to see that the tag
is unique, else raise an error.

#### Data Flow Analyses


##### Zero-R

Train and evaluate a Zero-R model using:

```sh
$ bazel run //deeplearning/ml4pl/models/zero_r -- \
    --graph_db="$DB?programl_${analysis}_graphs" \
    --log_db="$DB?programl_dataflow_logs" \
    --batch_scores_averaging_method=binary \
    --max_train_per_epoch=10000 \
    --max_val_per_epoch=1000 \
    --epoch_count=1 \
    --tag=dataflow_${analysis}_zero_r
```


##### LSTM

Train and evaluate an statement-level LLVM IR LSTM classifier using:

```sh
$ bazel run //deeplearning/ml4pl/models/lstm -- \
    --graph_db="$DB?programl_${analysis}_graphs" \
    --proto_db="$DB?programl_graph_protos" \
    --ir_db="$DB?programl_ir" \
    --log_db="$DB?programl_dataflow_logs" \
    --epoch_count=300 \
    --stop_at=val_acc=.9999,time=21600 \
    --padded_sequence_length=10000 \
    --padded_node_sequence_length=5000 \
    --batch_size=64 \
    --max_train_per_epoch=10000 \
    --max_val_per_epoch=10000 \
    --test_on=best \
    --batch_scores_averaging_method=binary \
    --tag=dataflow_${analysis}_lstm_ir
```

Useful configuration options are:

* `--padded_sequence_length=5000` The number of tokens to pad/truncate
  encoded text sequences to.
* `--padded_nodes_sequence_length=5000` The number of nodes to pad/truncate
  segmented encoded text sequences to.


##### GGNN

Train and evaluate a GGNN using:

```sh
$ bazel run //deeplearning/ml4pl/models/ggnn -- \
    --graph_db="$DB?programl_${analysis}_graphs" \
    --log_db="$DB?programl_dataflow_logs" \
    --epoch_count=300 \
    --stop_at=val_acc=.9999,time=21600 \
    --graph_batch_node_count=15000 \
    --max_node_count_limit_handler="skip" \
    --max_train_per_epoch=10000 \
    --max_val_per_epoch=10000 \
    --batch_scores_averaging_method=binary \
    --detailed_batch_types=test \
    --layer_timesteps=30 \
    --test_on=best \
    --tag=dataflow_${analysis}_ggnn
```

Useful configuration options are:

* `--graph_batch_node_count=15000` The maximum number of nodes to include in a
  disjoint batch of graphs.

#### Heterogeneous Device Mapping

##### Zero-R

```sh
$ bazel run //deeplearning/ml4pl/models/zero_r -- \
    --graph_db="$DB?programl_devmap_amd" \
    --log_db="$DB?programl_devmap_logs" \
    --k_fold \
    --epoch_count=1 \
    --tag=devmap_amd_zero_r
```

##### LSTM

Train an evaluate the LSTM_{OpenCL} model using:

```sh
$ bazel run //deeplearning/ml4pl/models/lstm -- \
    --graph_db="$DB?programl_devmap_amd" \
    --log_db="$DB?programl_devmap_logs" \
    --epoch_count=50 \
    --padded_sequence_length=1024 \
    --ir2seq=opencl \
    --test_on=improvement_and_last \
    --k_fold \
    --epoch_count=1 \
    --tag=devmap_amd_lstm_opencl
```

Train an evaluate the LSTM_{IR} model using:

```sh
$ bazel run //deeplearning/ml4pl/models/lstm -- \
    --graph_db="$DB?programl_devmap_amd" \
    --log_db="$DB?programl_devmap_logs" \
    --epoch_count=50 \
    --padded_sequence_length=15000 \
    --ir2seq=llvm \
    --test_on=improvement_and_last \
    --k_fold \
    --epoch_count=1 \
    --tag=devmap_amd_lstm_ir
```

```sh
$ bazel run //deeplearning/ml4pl/models/lstm -- \
    --graph_db="$DB?programl_devmap_amd" \
    --log_db="$DB?programl_devmap_logs" \
    --epoch_count=50 \
    --padded_sequence_length=15000 \
    --ir2seq=inst2vec \
    --test_on=improvement_and_last \
    --k_fold \
    --epoch_count=1 \
    --tag=devmap_amd_lstm_inst2vec
```

##### GGNN

```sh
$ bazel run //deeplearning/ml4pl/models/ggnn -- \
    --graph_db="$DB?programl_devmap_amd" \
    --log_db="$DB?programl_devmap_logs" \
    --graph_batch_node_count=15000 \
    --k_fold \
    --tag=devmap_amd_ggnn
```

### Step 5: Analyze Results

Export CSV files from a log database using:

```sh
$ bazel run //deeplearning/ml4pl/models:export_logs -- \
    --log_db="$DB?programl_logs" \
    --csv_dir=/tmp/logs/csvs
```

### Debugging

Many of the utility libraries are executable as binaries to produce data useful
for debugging.

#### Print Stats of a database

A database of intermediate representations:

```sh
$ bazel run //deeplearning/ml4pl/ir:ir_database -- \
    --ir_db="$DB?programl_ir"
```

A database of unlabelled program graphs:

```sh
$ bazel run //deeplearning/ml4pl/graphs/unlabelled:unlabelled_graph_database -- \
    --proto_db="$DB?programl_graph_protos"
```

A labelled graph database:

```sh
$ bazel run //deeplearning/ml4pl/graphs/labelled:graph_tuple_database -- \
    --graph_db="$DB?programl_graph_tuples"
```

A log database:

```sh
$ bazel run //deeplearning/ml4pl/models:log_database -- \
    --log_db="$DB?programl_logs"
```

#### Dump pickled graph tuples from a database

Read a database of graphs and dump to pickled graph tuples:

```sh
$ bazel run //deeplearning/ml4pl/graphs/labelled:graph_database_reader -- \
    --graph_db="$DB?programl_graph_tuples" \
    --graph_reader_outdir="$HOME/programl/graph_tuples"
```

#### Dump pickled graph tuple batches from a database

Assemble batches of disjoint graphs and dump to pickled graph tuples:

```sh
$ bazel run //deeplearning/ml4pl/graphs/labelled:graph_batcher -- \
    --graph_db="$DB?programl_graph_tuples" \
    --graph_batch_outdir="$HOME/programl/graph_batches"
```

#### Visualizing graphs

Create graphviz graphs from a pickled graph tuple:

```sh
$ bazel run //deeplearning/ml4pl/graphs/labelled:graph_tuple_viz -- \
    --graph_tuple="$HOME/programl/graph_tuples/graph_tuple_00001.pickle" \
    > $HOME/programl/graph_tuples/graph_tuple_00001.dot
```

## Contributing

See [CONTRIBUTING.md](/CONTRIBUTING.md).

## License

Copyright 2019-2020 the ProGraML authors and released under the terms of the
Apache Version 2.0 license. See LICENSE.

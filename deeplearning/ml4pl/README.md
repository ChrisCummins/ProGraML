# Graph-based Machine Learning for Programming Languages

## Setup

### Requirements

* Ubuntu >= 16.04 or macOS >= 10.14.
* Python >= 3.6.
* Bazel >= 0.28.0.
* MySQL >= 5.7.

### Installation

```sh
$ ./configure
# Answer yes/no questions. The defaults should be fine.
```

Run the test suite:

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
$ export DB="$HOME/mysql.txt"
$ test -f "$DB" || echo "wrong path"
```

## Usage

### Step 1: Gather Intermediate Representations

Generate `sqlite:////tmp/ir.db`, TODO. This is currently being refactored, and the existing databases have been
migrated from an old schema.


### Step 2: Generate Program Graphs

Generate `sqlite:////tmp/graph_protos.db`, TODO. This is currently being refactored, and the existing databases have been
migrated from an old schema.


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
 
 * `reachability` Control flow reachability.
 * `domtree` Dominator trees.
 * `liveness` Live-out variables.
 * `datadep` Data dependencies.
 * `subexpressions` Global common subexpressions.
 * `alias_set` Pointer alias sets.
 * `polyhedra` Polyhedral SScoP regions.

#### Heterogeneous Device Mapping

We provide a dataset of heterogeneous device mapping benchmarks for two devices:
`nvidia_gtx_960` and `amd_tahiti_7970`. Generate the labelled datasets using:

```sh
$ bazel run //deeplearning/ml4pl/graphs/labelled/devmap:make_devmap_dataset -- \
    --ir_db="$DB?programl_ir" \
    --proto_db="$DB?programl_graph_protos" \
    --graph_db="$DB?programl_devmap_nvidia" \
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

* `--test_on=improvement`: One of `{improvement,every}`
* `--keep_checkpoints=all`: One of `{all,last}`
* TODO (not yet implemented, all detailed batches are stored): 
`--detailed_batch_types=val,test`: A list of epoch types to make detailed batch 
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
    --graph_db="$DB?programl_reachability_graphs" \
    --log_db="$DB?programl_logs"
```

##### LSTM

Train and evaluate a Zero-R model using:

```sh
$ bazel run //deeplearning/ml4pl/models/lstm -- \
    --graph_db="$DB?programl_reachability_graphs" \
    --ir_db="$DB?programl_ir" \
    --log_db="$DB?programl_logs" \
    --nodes=statement
```

Useful configuration options are:

* `--padded_sequence_length=25000` The number of tokens to pad/truncate 
  sequences to.

##### GGNN

Train and evaluate a GGNN using:

```sh
$ bazel run //deeplearning/ml4pl/models/ggnn -- \
    --graph_db="$DB?programl_reachability_graphs" \
    --log_db="$DB?programl_logs" \
    --graph_batch_node_count=15000 \
    --layer_timesteps=30
```

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
    --tag=devmap_amd_lstm_llvm
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

Create graphviz graph from a pickled graph tuple:

```sh
$ bazel run //deeplearning/ml4pl/graphs/labelled:graph_tuple_viz -- \
    --graph_tuple="$HOME/programl/graph_tuples/graph_tuple_00001.pickle" \
    > $HOME/programl/graph_tuples/graph_tuple_00001.dot
```

### Contributing

Pull requests and bug reports welcome. If modifying or adding code, please add
tests. The most helpful way to report a bug is to submit a pull request which
adds a failing test case that reproduces the bug.

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
$ cat /tmp/mysql.txt
mysql://user:pass@hostname/
```


## Usage

### Step 1: Generate intermediate representations

Generate `sqlite:////tmp/ir.db`, TODO. This is currently being refactored, and the existing databases have been
migrated from an old schema.


### Step 2: Generate program graphs

Generate `sqlite:////tmp/program_graphs.db`, TODO. This is currently being refactored, and the existing databases have been
migrated from an old schema.


### Step 3: Annotate program graphs labels

#### Data flow analyses

Generate a database of graphs labelled with data flow analyses using:

```sh
$ bazel run //deeplearning/ml4pl/graphs/labelled:make_data_flow_analysis_dataset -- \
    --input_db='sqlite:////tmp/program_graphs.db' \
    --graph_db='sqlite:////tmp/reachability_graphs.db' \
    --analysis=reachability
```

### Step 4: Train a model

TODO. The model scripts need refactoring.

#### Zero-R

Train a Zero-R model using:

```sh
$ bazel run //deeplearning/ml4pl/models/zero_r -- \
    --graph_db='sqlite:////tmp/reachability_graphs.db' \
    --log_db='sqlite:////tmp/logs.db'
```

### Debugging

Many of the utility libraries are executable as binaries to produce data useful
for debugging.

#### Print stats of a graph tuple database

```sh
$ bazel run //deeplearning/ml4pl/graphs/labelled:graph_tuple_database -- \
    --graph_db='sqlite:////tmp/graph_tuples.db'
```

#### Dump pickled graph tuples from a database

Read a database of graphs and dump to pickled graph tuples:

```sh
$ bazel run //deeplearning/ml4pl/graphs/labelled:graph_database_reader -- \
    --graph_db='sqlite:////tmp/graph_tuples.db' \
    --graph_reader_outdir='/tmp/graph_tuples'
```

#### Dump pickled graph tuple batches from a database

Assemble batches of disjoint graphs and dump to pickled graph tuples:

```sh
$ bazel run //deeplearning/ml4pl/graphs/labelled:graph_batcher -- \
    --graph_db='sqlite:////tmp/graph_tuples.db' \
    --graph_batch_outdir='/tmp/graph_batches'
```

#### Visualizing graphs

Create graphviz graph from a pickled graph tuple:

```sh
$ bazel run //deeplearning/ml4pl/graphs/labelled:graph_tuple_viz -- \
    --graph_tuple='/tmp/graph_tuples/graph_tuple_00001.pickle' \
    > /tmp/graph_tuple.dot
```

### Contributing

Pull requests and bug reports welcome. If modifying or adding code, please add
tests. The most helpful way to report a bug is to submit a pull request which
adds a failing test case that reproduces the bug.

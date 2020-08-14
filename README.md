# ProGraML: Program Graphs for Machine Learning

<!-- license -->
<a href="https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg?color=brightgreen">
</a>
<!-- Commit counter -->
<a href="https://github.com/ChrisCummins/ProGraML/graphs/commit-activity">
  <img src="https://img.shields.io/github/commit-activity/y/ChrisCummins/ProGraML.svg?color=yellow">
</a>
<!-- Repo size -->
<a href="https://github.com/ChrisCummins/ProGraML">
  <img src="https://img.shields.io/github/repo-size/ChrisCummins/ProGraML.svg">
</a>

<!-- Build status -->
| Branch | Travis CI | Better Code |
| ------ | --------- | ----------- |
| [stable](https://github.com/ChrisCummins/ProGraML/tree/stable) | <a href="https://travis-ci.org/ChrisCummins/ProGraML"><img src="https://img.shields.io/travis/ChrisCummins/ProGraML/stable.svg"></a> | <a href="https://bettercodehub.com/results/ChrisCummins/ProGraML"><img src="https://bettercodehub.com/edge/badge/ChrisCummins/ProGraML?branch=stable"></a> |
| [development](https://github.com/ChrisCummins/ProGraML/tree/development) | <a href="https://travis-ci.org/ChrisCummins/ProGraML"><img src="https://img.shields.io/travis/ChrisCummins/ProGraML/development.svg"></a> | <a href="https://bettercodehub.com/results/ChrisCummins/ProGraML"><img src="https://bettercodehub.com/edge/badge/ChrisCummins/ProGraML?branch=development"></a> |


ProGraML is a representation for programs as input to a machine learning model.

Key features are:

* **Expressiveness:** We represent programs as graphs, capturing all of the
  control, data, and call relations. Each node in the graph represents an
  instruction, variable, or constant, and edges are positional such that
  non-commutative operations can be differentiated.
* **Portability:** ProGraML is derived from compiler IRs, making it independent
  of the source language (e.g. we have trained models to reason across five
  different source languages at a time). It is easy to target new IRs (we
  currently support [LLVM](/programl/Documentation/cmd/llvm2graph.txt) and
  [XLA](/programl/Documentation/cmd/xla2graph.txt)).
* **Extensibility:** Features and labels can easily be added at the
  whole-program level, per-instruction level, or for individual relations.


## Getting Started

To get stuck in and play around with our graph representation, visit:

[![Program Explorer](/programl/Documentation/assets/program_explorer.png)](https://chriscummins.cc/s/program_explorer)

Or if papers are more your ☕, have a read of ours:

[![Preprint](/programl/Documentation/arXiv.2003.10536/paper.png)](https://arxiv.org/abs/2003.10536)


## Constructing the ProGraML Representation

Here's a little example of producing the ProGraML representation for a simple
recursive Fibonacci implementation in C.

#### Step 1: Compiler IR

<img src="/programl/Documentation/assets/llvm2graph-1-ir.png" width=300>

We start by lowering the program to a compiler IR. In this case, we'll use
LLVM-IR. This can be done using: `clang -emit-llvm -S -O3 fib.c`.

#### Step 2: Control-flow

<img src="/programl/Documentation/assets/llvm2graph-2-cfg.png" width=300>

We begin building a graph by constructing a full-flow graph of the program. In a
full-flow graph, every instruction is a node and the edges are control-flow.
Note that edges are positional so that we can differentiate the branching
control flow in that `switch` instruction.

#### Step 3: Data-flow

<img src="/programl/Documentation/assets/llvm2graph-3-dfg.png" width=300>

Then we add a graph node for every variable and constant. In the drawing above,
the octagons are constants and the variables are ovals. We add data-flow edges
to describe the relations between constants and the instructions that use them,
and variables and the constants which define/use them. Like control edges, data
edges have positions. In the case of data edges, the position encodes the order
of a data element in the list of instruction operands.

#### Step 4: Call graph

<img src="/programl/Documentation/assets/llvm2graph-4-cg.png" width=300>

Then we add call edges (green) from callsites to the function entry
instruction, and return edges from function exits to the callsite. Since this
is a graph of a recursive function, the callsites refer back to the entry of
the function (the `switch`). The `external` node is used to represent a call
from an external site.

#### Step 5: Type graph

<img src="/programl/Documentation/assets/llvm2graph-5-types.png" width=300>

Finally, we add data types to the graph. Each unique type is represented as
a node and connected to all instances of this type through type edges.
Composite types such as structs, arrays, or pointers to types (not shown here)
are composed of individual nodes for each primitive type that are then composed
through type edges. For struct types, a numeric edge position indicates the
position of each element in the struct's field list.

The process described above can be run locally using our
[`clang2graph`](/programl/Documentation/cmd/clang2graph.txt) and
[`graph2dot`](/programl/Documentation/cmd/graph2dot.txt) tools: `clang
clang2graph -O3 fib.c | graph2dot`


## Datasets

Please see [this doc](/programl/Documentation/DataflowDataset.md) for download
links for our publicly available datasets of LLVM-IRs, ProGraML graphs, and data
flow analysis labels.


## Running the code


### Requirements

* macOS ≥ 10.15 or GNU / Linux (we recommend Ubuntu Linux ≥ 18.04).
* bazel ≥ 2.0 (we recommend using
  [bazelisk](https://github.com/bazelbuild/bazelisk) to automatically
  download and use the correct bazel version).
* Python ≥ 3.6
* MySQL client (N.B. this is not the full MySQL server, just the connector)
  * On macOS: `brew install mysql-client`
  * On Ubuntu: `sudo apt-get install libmysqlclient-dev`
* A Fortran compiler:
  * On macOS: `brew cask install gfortran`
  * (Ubuntu has one by default)
* (Optional) NVIDIA GPU with CUDA drivers for TensorFlow and PyTorch

Once you have the above requirements installed, test that everything is working
by building and running full test suite:

```sh
$ bazel test //programl/...
```


### Command-line tools

In the manner of Unix Zen, creating and manipulating ProGraML graphs is done
using [command-line tools](/programl/Documentation/cmd) which act as filters,
reading in graphs from stdin and emitting graphs to stdout. The structure for
graphs is described through a series of [protocol
buffers](/programl/Documentation/ProtocolBuffers.md).

Build and install the command line tools to `~/.local/opt/programl` (or a
directory of your choice) using:

```sh
$ bazel run -c opt //:install ~/.local/opt/programl
```

Then to use them, append the following to your `~/.bashrc`:

```sh
export PATH=~/.local/opt/programl/bin:$PATH
export LD_LIBRARY_PATH=~/.local/opt/programl/lib:$LD_LIBRARY_PATH
```


### Dataflow experiments

Download and unpack our [dataflow
dataset](/programl/Documentation/DataflowDataset.md), then train and evaluate a
graph neural network model using:

```sh
bazel run //programl/task/dataflow:train_ggnn \
    --analysis reachability \
    --path=$HOME/programl
```

where `--analysis` is the name of the analysis you want to evaluate, and
`--path` is the root of the unpacked dataset. There are a lot of options that
you can use to control the behavior of the experiment, see `--helpfull` for a
full list. Some useful ones include:

* `--batch_size` controls the number of nodes in each batch of graphs.
* `--layer_timesteps` defines the layers of the GGNN model, and the number of timesteps used for
  each.
* `--learning_rate` sets the initial learning rate of the optimizer.
* `--lr_decay_rate` the rate at which learning rate decays.
* `--lr_decay_steps` number of gradient steps until the lr is decayed.
* `--train_graph_counts` lists the number of graphs to train on between runs of the validation set.


### Using this project as a dependency

If you are using bazel you can add ProGraML as an external dependency. Add to
your WORKSPACE file:

```py
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name="programl",
    strip_prefix="programl-<stable-commit>",
    urls=["https://github.com/ChrisCummins/labm8/archive/<stable-commit>.tar.gz"],
)

# === Begin ProGraML dependencies ===
<WORKSPACE dependencies>
# === End ProGraML dependencies ===
```

Where `<WORKSPACE dependencies>` is the block of delimited code in
[@programl//:WORKSPACE](https://github.com/ChrisCummins/ProGraML/blob/development/WORKSPACE)
(this is an unfortunately clumsy workaround for [recursive
workspaces](https://github.com/bazelbuild/bazel/issues/1943)).

Then in your BUILD file:

```py
cc_library(
    name = "mylib",
    srcs = ["mylib.cc"],
    deps = [
        "@programl//programl/ir/llvm",
    ],
)

py_binary(
    name = "myscript",
    srcs = ["myscript.py"],
    deps = [
        "@programl//programl/ir/llvm/py:llvm",
    ],
)
```


## Contributing

Patches, bug reports, feature requests are welcome! Please use the
[issue tracker](https://github.com/ChrisCummins/ProGraML/issues) to file a
bug report or question. Please read the
[development workflow](/programl/Documentation/Development.md)
document before contributing code.


## Acknowledgements

Made with ❤️️ by
[Chris Cummins](https://chriscummins.cc/) and
[Zach Fisches](https://github.com/Zacharias030),
with help from folks at the University of Edinburgh and ETH Zurich:
[Tal Ben-Nun](https://people.inf.ethz.ch/tbennun/),
[Torsten Hoefler](https://htor.inf.ethz.ch/),
[Hugh Leather](http://homepages.inf.ed.ac.uk/hleather/), and
[Michael O'Boyle](http://www.dcs.ed.ac.uk/home/mob/).

Funding sources: [HiPEAC Travel Grant](https://www.hipeac.net/collaboration-grants/).

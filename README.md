# ProGraML: Program Graphs for Machine Learning

<!-- Build status -->
<table>
    <tr>
      <td>License</td>
      <td>
        <a href="https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)">
          <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg?color=brightgreen">
        </a>
      </td>
    </tr>
    <tr>
      <td>OS</td>
      <td>GNU/Linux, macOS ≥ 10.15</td>
    </tr>
    <tr>
      <td>Python Versions</td>
      <td>3.6, 3.7, 3.8, 3.9</td>
    </tr>
    <tr>
      <td>
        <a href="https://github.com/ChrisCummins/ProGraML/tree/development">
          development
        </a> Branch
      </td>
      <td>
        <a href="https://github.com/ChrisCummins/ProGraML/actions?query=workflow%3ACI+branch%3Adevelopment">
          <img src="https://github.com/ChrisCummins/ProGraML/workflows/CI/badge.svg?branch=development">
        </a>
        <a href="https://bettercodehub.com/results/ChrisCummins/ProGraML">
          <img src="https://bettercodehub.com/edge/badge/ChrisCummins/ProGraML?branch=development">
        </a>
      </td>
    </tr>
      <tr>
      <td>
        <a href="https://github.com/ChrisCummins/ProGraML/tree/stable">
          stable
        </a> Branch
      </td>
      <td>
        <a href="https://github.com/ChrisCummins/ProGraML/actions?query=workflow%3ACI+branch%3Astable">
          <img src="https://github.com/ChrisCummins/ProGraML/workflows/CI/badge.svg?branch=stable">
        </a>
        <a href="https://bettercodehub.com/results/ChrisCummins/ProGraML">
          <img src="https://bettercodehub.com/edge/badge/ChrisCummins/ProGraML?branch=stable">
        </a>
      </td>
    </tr>
    <tr>
      <td>Development Activity</td>
      <td>
        <a href="https://github.com/ChrisCummins/ProGraML/graphs/commit-activity">
          <img src="https://img.shields.io/github/commit-activity/y/ChrisCummins/ProGraML.svg?color=yellow">
        </a>
        <a href="https://github.com/ChrisCummins/ProGraML">
          <img src="https://img.shields.io/github/repo-size/ChrisCummins/ProGraML.svg">
        </a>
      </td>
    </tr>
</table>


<!-- MarkdownTOC autolink="true" -->

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Installation](#installation)
    - [Command-line tools](#command-line-tools)
    - [Building from source](#building-from-source)
    - [Datasets](#datasets)
    - [Using this project as a dependency](#using-this-project-as-a-dependency)
- [Constructing the ProGraML Representation](#constructing-the-programl-representation)
    - [Step 1: Compiler IR](#step-1-compiler-ir)
    - [Step 2: Control-flow](#step-2-control-flow)
    - [Step 3: Data-flow](#step-3-data-flow)
    - [Step 4: Call graph](#step-4-call-graph)
- [Usage](#usage)
    - [End-to-end C++ flow](#end-to-end-c-flow)
    - [Dataflow experiments](#dataflow-experiments)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

<!-- /MarkdownTOC -->


## Overview


ProGraML is a representation for programs as input to a machine learning model.

Key features are:

* **Expressiveness:** We represent programs as graphs, capturing all of the
  control, data, and call relations. Each node in the graph represents an
  instruction, variable, or constant, and edges are positional such that
  non-commutative operations can be differentiated.
* **Portability:** ProGraML is derived from compiler IRs, making it independent
  of the source language (e.g. we have trained models to reason across five
  different source languages at a time). It is easy to target new IRs (we
  currently support [LLVM](/Documentation/bin/llvm2graph.txt) and
  [XLA](/Documentation/bin/xla2graph.txt)).
* **Extensibility:** Features and labels can easily be added at the
  whole-program level, per-instruction level, or for individual relations.


## Getting Started

To get stuck in and play around with our graph representation, visit:

[![Program Explorer](/Documentation/assets/program_explorer.png)](https://chriscummins.cc/s/program_explorer)

Or if papers are more your ☕, have a read of ours:

[![Preprint](/Documentation/arXiv.2003.10536/paper.png)](https://arxiv.org/abs/2003.10536)


## Installation


#### Command-line tools

1. Download the latest macOS or Linux release archive from the [releases page](https://github.com/ChrisCummins/ProGraML/releases).
2. Unpack the release archive to `~/.local/opt/programl` (or a directory of your choice) using:
```sh
mkdir -p ~/.local/opt/programl
tar xjvf ~/Downloads/programl-*.tar.bz2 -C ~/.local/opt/programl
```
3. Add the installed files to your paths. You may want to add this to your `~/.bashrc`:
```sh
export PATH=$HOME/.local/opt/programl/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/opt/programl/lib:$LD_LIBRARY_PATH
```

#### Building from source

Requirements:

* macOS ≥ 10.15 or GNU / Linux (we recommend Ubuntu Linux ≥ 18.04).
* bazel ≥ 2.0 (we recommend using
  [bazelisk](https://github.com/bazelbuild/bazelisk) to automatically
  download and use the correct bazel version).
* Python ≥ 3.6

Install the python dependencies using:

```
$ python -m pip install -r requirements.txt
```

Once you have the above requirements installed, test that everything is working
by building and running full test suite:

```sh
$ bazel test //...
```

Build and install the command line tools to `~/.local` (or a
directory of your choice) using:

```sh
$ bazel run -c opt //:install -- ~/.local
```

Then to use them, append the following to your `~/.bashrc`:

```sh
export PATH=~/.local/opt/programl/bin:$PATH
export LD_LIBRARY_PATH=~/.local/opt/programl/lib:$LD_LIBRARY_PATH
```


#### Datasets

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4122437.svg)](https://doi.org/10.5281/zenodo.4122437)

Please see [this doc](/Documentation/DataflowDataset.md) for download
links for our publicly available datasets of LLVM-IRs, ProGraML graphs, and data
flow analysis labels.


#### Using this project as a dependency

If you are using bazel you can add ProGraML as an external dependency. Add to
your WORKSPACE file:

```py
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name="programl",
    strip_prefix="ProGraML-<stable-commit>",
    urls=["https://github.com/ChrisCummins/ProGraML/archive/<stable-commit>.tar.gz"],
)

# ----------------- Begin ProGraML dependencies -----------------
<WORKSPACE dependencies>
# ----------------- End ProGraML dependencies -----------------
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


## Constructing the ProGraML Representation

The ProGraML representation is constructed in multiple stages. Here we describe
the process for a simple recursive Fibonacci implementation in C. For
instructions on how to run this process, see [Usage](#usage) below.


#### Step 1: Compiler IR

<img src="/Documentation/assets/llvm2graph-1-ir.png" width=300>

We start by lowering the program to a compiler IR. In this case, we'll use
LLVM-IR. This can be done using: `clang -emit-llvm -S -O3 fib.c`.


#### Step 2: Control-flow

<img src="/Documentation/assets/llvm2graph-2-cfg.png" width=300>

We begin building a graph by constructing a full-flow graph of the program. In a
full-flow graph, every instruction is a node and the edges are control-flow.
Note that edges are positional so that we can differentiate the branching
control flow in that `switch` instruction.


#### Step 3: Data-flow

<img src="/Documentation/assets/llvm2graph-3-dfg.png" width=300>

Then we add a graph node for every variable and constant. In the drawing above,
the diamonds are constants and the variables are ovals. We add data-flow edges
to describe the relations between constants and the instructions that use them,
and variables and the constants which define/use them. Like control edges, data
edges have positions. In the case of data edges, the position encodes the order
of a data element in the list of instruction operands.


#### Step 4: Call graph

<img src="/Documentation/assets/llvm2graph-4-cg.png" width=300>

Finally, we add call edges (green) from callsites to the function entry
instruction, and return edges from function exits to the callsite. Since this is
a graph of a recursive function, the callsites refer back to the entry of the
function (the `switch`). The `external` node is used to represent a call from an
external site.

The process described above can be run locally using our
[`clang2graph`](/Documentation/bin/clang2graph.txt) and
[`graph2dot`](/Documentation/bin/graph2dot.txt) tools: `clang
clang2graph -O3 fib.c | graph2dot`


## Usage


#### End-to-end C++ flow

In the manner of Unix Zen, creating and manipulating ProGraML graphs is done
using [command-line tools](/Documentation/bin) which act as filters,
reading in graphs from stdin and emitting graphs to stdout. The structure for
graphs is described through a series of [protocol
buffers](/Documentation/ProtocolBuffers.md).

This section provides an example step-by-step guide for generating a program
graph for a C++ application.

1. Install LLVM-10 and the ProGraML [command line tools](#command-line-tools).
2. Compile your C++ code to LLVM-IR. The way to do this to modify your build
   system so that clang is passed the `-emit-llvm -S` flags. For a
   single-source application, the command line invocation would be:
```
$ clang-10 -emit-llvm -S -c my_app.cpp -o my_app.ll
```
   For a multi-source application, you can compile each file to LLVM-IR
   separately and then link the results. For example:
```
$ clang-10 -emit-llvm -S -c foo.cpp -o foo.ll
$ clang-10 -emit-llvm -S -c bar.cpp -o bar.ll
$ llvm-link foo.ll bar.ll -S -o my_app.ll
```
3. Generate a ProGraML graph protocol buffer from the LLVM-IR using the
   [llvm2graph](https://github.com/ChrisCummins/ProGraML/blob/development/Documentation/bin/llvm2graph.txt)
   commnand:
```
$ llvm2graph < my_app.ll > my_app.pbtxt
```
   The generated file `my_app.pbtxt` uses a human-readable
   [ProgramGraph](https://github.com/ChrisCummins/ProGraML/blob/development/programl/proto/program_graph.proto)
   format which you can inspect using a text editor. In this case, we will
   render it to an image file using Graphviz.

4. Generate a Graphviz dotfile from the ProGraML graph using
   [graph2dot](https://github.com/ChrisCummins/ProGraML/blob/development/Documentation/bin/graph2dot.txt):
```
$ graph2dot < my_app.pbtxt > my_app.dot
```
5. Render the dotfile to a PNG image using Graphviz:
```
$ dot -Tpng my_app.dot -o my_app.png
```


#### Dataflow experiments

1. Follow the instructions for [building from source](#building-from-source)
2. Download and unpack our [dataflow
dataset](/Documentation/DataflowDataset.md)
3. Train and evaluate a graph neural network model using:

```sh
bazel run -c opt //tasks/dataflow:train_ggnn -- \
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

🏗️ **Under construction** We are in the process of refactoring the dataflow
experiments with a revamped API.  There are currently bugs in the data loader
which may affect training jobs, see
[#147](https://github.com/ChrisCummins/ProGraML/issues/147).


## Contributing

Patches, bug reports, feature requests are welcome! Please use the
[issue tracker](https://github.com/ChrisCummins/ProGraML/issues) to file a
bug report or question. If you would like to help out with the code, please
read [this document](CONTRIBUTING.md).


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

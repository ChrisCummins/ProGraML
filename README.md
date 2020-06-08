# ProGraML: Program Graphs for Machine Learning

<!-- Travis CI -->
<a href="https://travis-ci.org/ChrisCummins/ProGraML">
  <img src="https://img.shields.io/travis/ChrisCummins/ProGraML/master.svg">
</a>
<!-- Better code -->
<a href="https://bettercodehub.com/results/ChrisCummins/ProGraML">
  <img src="https://bettercodehub.com/edge/badge/ChrisCummins/ProGraML?branch=master">
</a>
<!-- license -->
<a href="https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg?color=brightgreen">
</a>
<!-- Repo size -->
<a href="https://github.com/ChrisCummins/ProGraML">
  <img src="https://img.shields.io/github/repo-size/ChrisCummins/ProGraML.svg">
</a>
<!-- Commit counter -->
<a href="https://github.com/ChrisCummins/ProGraML/graphs/commit-activity">
  <img src="https://img.shields.io/github/commit-activity/y/ChrisCummins/ProGraML.svg?color=yellow">
</a>

ProGraML is a representation for programs as input to a machine learning model.

Key features are:

* **Expressiveness:** We represent programs as graphs, capturing all of the control, data, and call
  relations. Each node in the graph represents an instruction, variable, or constant, and edges
  are positional such that non-commutative operations can be differentiated.
* **Portability:** ProGraML is derived from compiler IRs, making it independent of the source
  language (e.g. we have trained models to reason across five different source languages at a time).
  It is easy to target new IRs (we currently support [LLVM](/programl/Documentation/cmd/llvm2graph.txt) and
  [XLA](/programl/Documentation/cmd/xla2graph.txt)).
* **Extensibility:** Features and labels can easily be added at the whole-program level, per-instruction level,
  or for individual relations.


## Getting Started

To get stuck in and play around with our graph representation, visit:

[![Program Explorer](/programl/Documentation/assets/program_explorer.png)](https://chriscummins.cc/s/program_explorer)

Or if papers are more your ☕, have a read of ours:

[![Preprint](/programl/Documentation/arXiv.2003.10536/paper.png)](https://arxiv.org/abs/2003.10536)


## Datasets

Please see [this doc](/programl/Documentation/DataflowDataset.md) for download links for our publicly available
datasets of LLVM-IRs, ProGraML graphs, and data flow analysis labels.


## Running the code


### Requirements

* macOS or GNU / Linux
* bazel >= 2.0
* Python >= 3.6
* (Optional) NVIDIA GPU with CUDA drivers for TensorFlow and PyTorch

See [INSTALL](/INSTALL.md) for the lowdown on setting up your build environment. Once done,
you should be able to build and run the full test suite using:

```sh
$ bazel test //programl/...
```


### Command-line tools

In the manner of Unix Zen, creating and manipulating ProGraML graphs is done using
[command-line tools](/programl/Documentation/cmd) which act as filters, reading in graphs from stdin and
emitting graphs to stdout. The structure for graphs is described through a series of
[protocol buffers](/programl/Documentation/ProtocolBuffers.md).

Build and install the command line tools to `~/.local/opt/programl` (or a directory of your choice)
using:

```sh
$ bazel run -c opt //programl:install ~/.local/opt/programl
```

Then to use them, append the following to your `~/.bashrc`:

```sh
export PATH=~/.local/opt/programl/bin:$PATH
export LD_LIBRARY_PATH=~/.local/opt/programl/lib:$LD_LIBRARY_PATH
```


### Dataflow experiments

Download and unpack our [dataflow dataset](/programl/Documentation/DataflowDataset.md), then train and
evaluate a graph neural network model using:

```sh
bazel run //programl/task/dataflow:train_ggnn \
    --analysis reachability \
    --path=$HOME/programl
```

where `--analysis` is the name of the analysis you want to evaluate, and `--path` is the root of the
unpacked dataset. There are a lot of options that you can use to control the behavior of the
experiment, see `--helpfull` for a full list. Some useful ones include:

* `--batch_size` controls the number of nodes in each batch of graphs.
* `--layer_timesteps` defines the layers of the GGNN model, and the number of timesteps used for
  each.
* `--learning_rate` sets the initial learning rate of the optimizer.
* `--lr_decay_rate` the rate at which learning rate decays.
* `--lr_decay_steps` number of gradient steps until the lr is decayed.
* `--train_graph_counts` lists the number of graphs to train on between runs of the validation set.


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

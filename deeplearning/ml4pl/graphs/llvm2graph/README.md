# llvm2graph

This package constructs program graphs from LLVM intermediate representations.
The [`llvm::LlvmGraphBuilder`](deeplearning/ml4pl/graphs/llvm2graph/llvm_graph_builder.h)
class implements a graph constructor which accepts an `llvm::Module` as input
and generates a ProgramGraph protocol buffer. The
[`llvm2graph`](deeplearning/ml4pl/graphs/llvm2graph/llvm2graph.cc) binary
wraps the graph constructor in a pass pipeline and runs it on a module.

## Usage

```sh
$ bazel run //deeplearning/ml4pl/graphs/llvm2graph -- /path/to/ir.ll
```

Reads the given LLVM module and prints the program graph proto to stdout.

Alternatively, use `--stdout_format=dot` to print a graphviz graph:

```sh
$ bazel run //deeplearning/ml4pl/graphs/llvm2graph -- \
    $PWD/deeplearning/ml4pl/testing/data/bytecode_regression_tests/53.ll \
    --stdout_format=dot > /tmp/graph.dot
$ dot -Tpng /tmp/graph.dot -o /tmp/graph.png
```

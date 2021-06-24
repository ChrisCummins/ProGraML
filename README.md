<h1 align="center">ProGraML: Program Graphs for Machine Learning</h1>

<p align="center">
  <!-- PyPi Version -->
  <a href="https://pypi.org/project/programl/">
      <img src="https://badge.fury.io/py/programl.svg" alt="PyPI version" height="20">
  </a>
  <!-- Downloads counter -->
  <a href="https://pypi.org/project/programl/">
      <img src="https://pepy.tech/badge/programl" alt="PyPi Downloads" height="20">
  </a>
  <!-- license -->
  <a href="https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)">
      <img src="https://img.shields.io/pypi/l/programl" alt="License" height="20">
  </a>
  <!-- CI status -->
  <a href="https://github.com/ChrisCummins/ProGraML/actions?query=workflow%3ACI+branch%3Astable">
    <img src="https://github.com/ChrisCummins/ProGraML/workflows/CI/badge.svg?branch=stable">
  </a>
  <!-- Better code -->
  <a href="https://bettercodehub.com/results/ChrisCummins/ProGraML">
    <img src="https://bettercodehub.com/edge/badge/ChrisCummins/ProGraML?branch=stable">
  </a>
  <!-- Commit counter -->
  <a href="https://github.com/ChrisCummins/ProGraML/graphs/commit-activity">
    <img src="https://img.shields.io/github/commit-activity/y/ChrisCummins/ProGraML.svg?color=yellow">
  </a>
</p>

<p align="center">
  <i>An expressive, language-independent representation of programs.</i>
</p>
<p align="center">
  <i>
    Check <a href="https://chriscummins.cc/ProGraML">the website</a>
    for more information.
  </i>
</p>


## Introduction

ProGraML is a representation for programs as input to a machine learning model.
The key features are:

1. **Simple:** Everything is available through a `pip install`, no compilation
   required. Supports several programming languages (*C, C++, LLVM-IR, XLA*) and
   several graph formats (*NetworkX, DGL, Graphviz, JSON*) out of the box.

2. **Expressive:** Captures every control, data, and call relation across entire
   programs. The representation is independent of the source language. Features
   and labels can be added at any granularity to support whole-program,
   per-instruction, or per-relation reasoning tasks.

3. **Fast:** The core graph construction is implemented in C++ with a low
   overhead interface to Python. Every API method supports simple and efficient
   parallelization through an `executor` parameter.

To get stuck in and play around with our graph representation, visit:

<a href="https://chriscummins.cc/s/program_explorer">
  <img height="400" src="https://github.com/ChrisCummins/ProGraML/raw/development/Documentation/assets/program_explorer.png">
</a>

Or if papers are more your ☕, have a read of ours:

<a href="https://chriscummins.cc/pub/2021-icml.pdf">
  <img height="325" src="https://github.com/ChrisCummins/ProGraML/raw/development/Documentation/icml-2021/paper.png">
</a>


## Getting Started

Install the latest release of the Python package using:

```
pip install -U programl
```

The API is very simple, comprising graph *creation* ops, graph *transform* ops,
and graph *serialization* ops. Here is a quick demo of each:

```py
>>> import programl as pg

# Construct a program graph from C++:
>>> G = pg.from_cpp("""
... #include <iostream>
...
... int main(int argc, char** argv) {
...   std::cout << "Hello, world!" << std::endl;
...   return 0;
... }
... """)

# A program graph is a protocol buffer:
>>> type(G).__name__
'ProgramGraph'

# Convert the graph to NetworkX:
>>> pg.to_networkx(G)
<networkx.classes.multidigraph.MultiDiGraph at 0x7fbcf40a2fa0>

# Save the graph for later:
>>> pg.save_graphs('file.data', [G])
```

For further details check out the [API
reference](https://chriscummins.cc/ProGraML/api/python.html).

## Supported Programming Languages

The following programming languages and compiler IRs are supported
out-of-the-box:

<table>
  <tr>
    <th>Language</th>
    <th>API Calls</th>
    <th>Supported Versions</th>
  </tr>
  <tr>
    <td>C</td>
    <td>
      <a href="https://chriscummins.cc/ProGraML/api/python.html#programl.from_cpp"><code>programl.from_cpp()</code></a>,
      <a href="https://chriscummins.cc/ProGraML/api/python.html#programl.from_clang"><code>programl.from_clang()</code></a>
    </td>
    <td>Up to ISO C 2017</td>
  </tr>
  <tr>
    <td>C++</td>
    <td>
      <a href="https://chriscummins.cc/ProGraML/api/python.html#programl.from_cpp"><code>programl.from_cpp()</code></a>,
      <a href="https://chriscummins.cc/ProGraML/api/python.html#programl.from_clang"><code>programl.from_clang()</code></a>
    </td>
    <td>Up to ISO C++ 2020 DIS</td>
  </tr>
  <tr>
    <td>LLVM-IR</td>
    <td>
      <a href="https://chriscummins.cc/ProGraML/api/python.html#programl.from_llvm_ir"><code>programl.from_llvm_ir()</code></a>
    </td>
    <td>3.8.0, 6.0.0, 10.0.0</td>
  </tr>
  <tr>
    <td>XLA</td>
    <td>
      <a href="https://chriscummins.cc/ProGraML/api/python.html#programl.from_xla_hlo_proto"><code>programl.from_xla_hlo_proto()</code></a>
    </td>
    <td>2.0.0</td>
  </tr>
</table>

Is your favorite language not supported here? Submit a [feature
request](https://github.com/ChrisCummins/ProGraML/issues/new/choose)!


## Contributing

Patches, bug reports, feature requests are welcome! Please use the
[issue tracker](https://github.com/ChrisCummins/ProGraML/issues) to file a
bug report or question. If you would like to help out with the code, please
read [this document](CONTRIBUTING.md).


## Citation

If you use ProGraML in any of your work, please cite [this
paper](https://chriscummins.cc/pub/2021-icml.pdf):

```
@inproceedings{cummins2021a,
  title={{ProGraML: A Graph-based Program Representation for Data Flow Analysis and Compiler Optimizations}},
  author={Cummins, Chris and Fisches, Zacharias and Ben-Nun, Tal and Hoefler, Torsten and O'Boyle, Michael and Leather, Hugh},
  booktitle = {Thirty-eighth International Conference on Machine Learning (ICML)},
  year={2021}
}
```

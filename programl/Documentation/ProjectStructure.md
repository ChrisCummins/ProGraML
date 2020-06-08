# Project Structure

This repository is exported from a subset of my [phd](https://github.com/ChrisCummins/phd)
monorepo. The core of the project is rooted in the `programl` directory, everything else is
supporting libraries.

This project is divided into the following top level packages:

* [`programl/cmd`](/programl/cmd): Command line tools. All executable binaries are stored here.
* [`programl/Documentation`](/programl/Documentation): Additional (possibly out-of-date 😢) documentation.
* [`programl/graph`](/programl/graph): Libraries for creating and manipulating the ProGraML graph representation.
* [`programl/ir`](/programl/ir): Support for specific compiler IRs, e.g. LLVM.
* [`programl/proto`](/programl/proto): Protocol message definitions. These define the core data structures for this project.
  This is a good starting point for understanding the code.
* [`programl/task`](/programl/task): Experimental tasks. This contains the code for producing the results of our published
  experiments.
* [`programl/test`](/programl/test): Test data and helper libraries.
* [`programl/util`](/programl/util): Utility code.

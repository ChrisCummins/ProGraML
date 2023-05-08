# Project Structure

The key directories of this project are:

* [`/Documentation`](/Documentation): Additional (possibly out-of-date 😢)
  documentation.
* [`/programl/bin`](/programl/bin): Command line tools. All executable binaries
  are stored here.
* [`/programl/graph`](/programl/graph): Libraries for creating and manipulating
  the ProGraML graph representation.
* [`/programl/ir`](/programl/ir): Support for specific compiler IRs, e.g. LLVM.
* [`/programl/proto`](/programl/proto): Protocol message definitions. These
  define the core data structures for this project. This is a good starting
  point for understanding the code.
* [`/programl/util`](/programl/util): Utility code.
* [`/tasks`](/tasks): Experimental tasks. This contains the code for producing
  the results of our published experiments.
* [`/tests`](/tests): Tests.

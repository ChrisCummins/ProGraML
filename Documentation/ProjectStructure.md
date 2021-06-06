# Project Structure

The key directories of this project are:

* [`/bin`](/bin): Command line tools. All executable binaries are stored here.
* [`/Documentation`](/Documentation): Additional (possibly out-of-date 😢)
  documentation.
* [`/models`](/model): Model implementations and helper code.
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

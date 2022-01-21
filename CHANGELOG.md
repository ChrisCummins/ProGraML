## v0.3.2 (2022-01-21)

This release fixes a bug in which, for some LLVM IRs, the control edge position
values for instruction nodes with outdegree > 1 would be incorrectly set to 0.

## v0.3.1 (2021-10-15)

This micro release relaxes the version requirements of the protobuf and grpcio
Python dependencies, and adds torch to the core dependencies.

## v0.3.0 (2021-06-24)

This release adds a much simpler, flat Python API. The API comprises three
families of functions: graph *creation* ops, graph *transform* ops, and graph
*serialization* ops. Each function supports simple and efficient parallelization
through an `executor` parameter, and can be chained together.

This release also adds support for LLVM 6.0.0.

## v0.2.0 (2021-06-06)

Move model definitions out of the `programl` package and provide a prebuilt
wheel for the Python package.

## v0.1.0 (2020-12-17)

Initial release.

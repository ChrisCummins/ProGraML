# Third party Python packages

This directory contains "mock" packages for pulling in python `requirements`. We
do this to allow for packages which contain undocumented dependencies which must
be added (see `//third_party/py/progressbar` for an example), or to allow us to
group multiple packages under a single bazel target (see
`//third_party/py/pytest`).

## To add a package

1. Add the new pip packages to `//tools/requirements.txt`.
1. Create a package in this directory which contains a single `py_library` rule
   and pulls in the new pip package as a `dep` (copy any of the existing 
   packages as a starting point).
1. Add the `//third_party/py/<package>` dep to any python targets which require
   this new module.

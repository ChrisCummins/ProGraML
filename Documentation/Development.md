# Development

This document describes the development workflow for ProGraML.

## Branching Model

This project uses the
[git flow](https://danielkummer.github.io/git-flow-cheatsheet/) branching model
to ensure that changes to the graph data format which break compatibility are
recognized by different release versions.

Please write new features by branching off of `development` and using the
branch name format: `feature/<name>_<issue>`, where `<name>` is a short
snake_case summary of the feature, and `<issue>` is an issue number on the
[issue tracker](https://github.com/ChrisCummins/ProGraML/issues) (all new
features should start with a tracking issue).

To use git flow:

1. [Install git-flow](https://github.com/nvie/gitflow/wiki/Installation).
2. Initialize it in this repository by running:

```
$ git flow init

Which branch should be used for bringing forth production releases?
Branch name for production releases: stable

Which branch should be used for integration of the "next release"?
Branch name for "next release" development: development

How to name your supporting branch prefixes?
Feature branches? [feature/]
Release branches? [release/]
Hotfix branches? [hotfix/]
Support branches? [support/]
Version tag prefix? []
```

## Commits

Try to ensure that commits are small and atomic. Commit messages should
describe:

1. The goal of the change.
2. Example usage of the new code (if appropriate).
3. What you did to test the change (if not covered by existing tests).
4. The URL of the GitHUb tracking issue,
   e.g., `github.com/ChrisCummins/ProGraML/issues/76`.

Example:

```
From 6aa9b5bbec0525b27c950d59b773e1585cedf4a3 Mon Sep 17 00:00:00 2001
From: Chris Cummins <chrisc.101@gmail.com>
Date: Fri, 21 Aug 2020 22:45:06 +0100
Subject: [PATCH] Update dependencies: networkx, scipy, scikit-learn.

Python 3.8 introduced some failures in the python pypi dependencies:

scipy:

    $ bazel test //programl/models:rolling_results_test

    ImportError: dlopen(${bazelroot}/bin/programl/test/benchmarks/benchmark_dataflow_analyses.runfiles/programl_requirements_pypi__scipy_1_2_1/scipy/linalg/cython_lapack.cpython-38-darwin.so, 2): Symbol not found: _cbbcsd_
      Referenced from: ${bazelroot}/bin/programl/test/benchmarks/benchmark_dataflow_analyses.runfiles/programl_requirements_pypi__scipy_1_2_1/scipy/linalg/cython_lapack.cpython-38-darwin.so
      Expected in: flat namespace

scikit-learn:

    $ bazel test //programl/test/benchmarks:benchmark_dataflow_analyses

      File "${bazelroot}/bin/programl/models/rolling_results_test.runfiles/programl_requirements_pypi__scikit_learn_0_20_3/sklearn/externals/joblib/externals/cloudpickle/cloudpickle.py", line 148, in _make_cell_set_template_code
        return types.CodeType(
    TypeError: an integer is required (got type bytes)

networkx:

    $ bazel test //programl/ir/llvm:inst2vec_test

    ${bazelroot}/bin/programl/cmd/inst2vec.runfiles/programl_requirements_pypi__networkx_2_2/networkx/drawing/nx_pydot.py:210: SyntaxWarning: "is" with a literal. Did you mean "=="?

This patch updates those packages to the versions to maintain support on
Python 3.8. Tested on macOS 10.15, python 3.8.

github.com//issues/76
---
 requirements.txt                  | 7 ++++---
 third_party/py/scikit_learn/BUILD | 1 +
 2 files changed, 5 insertions(+), 3 deletions(-)
```

## Tests

Please, please, test new code.

To paraphrase Kanye West, "one good test is worth a thousand patches". Use
bazel's excellent testing infrastructure to run the existing tests, and the new
ones that you write). Although bazel caches test results and will perform only
the minimum amount of testing on incremental builds, this is a large project
with a lot of tests, you may not want to run all of them as many will not be
relevant to what you are working on. Use the CI script provided by bazel to run
the tests on only those targets which have been modified on your current branch:

```sh
$ ./third_party/bazel/ci.sh
```

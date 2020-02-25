# Installation

This document describes the requirements and process for building the code
in this project, along with a list of known issues. Please read through this
document first if you are encountering problems.


## 1. Requirements

This project requires a modern version of Ubuntu Linux or macOS. The core
dependencies of the project are:

* [Bazel](https://docs.bazel.build/versions/master/install.html) >= 2.0.0.
* Python >= 3.6.
* A modern C++ toolchain with support for (at least) C++14.

#### Building with Docker

I maintain a docker image
[chriscummins/phd_build](https://hub.docker.com/r/chriscummins/phd_build)
which is a pre-configured Linux distribution with all build dependencies for
this project. If you are using an unsupported system, or if you don't have the
necessary privileges to install the depenencies, you can use this image to run
all of your builds and tests:

```sh
$ PHD="$(pwd)" bash tools/docker/phd_build/run.sh
```

This will drop you into a bash prompt from which you can run builds and tests,
saving you the need to install any dependencies on your host system. Note that
this image is large (sorry about that, LaTeX distributions are huge), and the
symlinks that bazel creates for `bazel-bin` and the like may be replaced.


#### Additional dependencies

Many sub-projects have additional build-time or runtime dependencies, such as
docker, Java, or go. On a supported system, run the following script to generate
a script which exhaustively installs all dependencies:

``` sh
$ ./tools/make_bootstrap.sh
```

This creates a file `bootstrap.sh`. Inspect this script, and if it looks
good to you, run it.


## 2. Building the code

This project uses the bazel build system. I'm a big fan of bazel, but there's
no denying that it has some quirks. If you are unfamiliar with it, I would
suggest having a quick flick through
[the docs](https://docs.bazel.build/versions/master/build-ref.html) to
understand the basic terminology and design decisions, it's probably unlike any
build system you have used before. The core commands you will use are `run` and
`test`.

#### bazel run //path/to:target

This builds the specified target and executes it. A few things to be aware of:

* Command line flags which you intend to pass to the executable need to be
  escaped with `--` to prevent bazel from attempting to interpret them, e.g.
  `bazel run //path/to:target -- --flag=foo`.
* Almost every executable target in this project provides documentation and
  usage information using the `--help` flag. Try it!
  `bazel run //path/to:target -- --help`.
* Bazel defaults to a debug build for quicker compiles. To build an optimized
  executable, use the `-c opt` option, e.g. `bazel run -c opt //path/to:target`.
* Bazel ignores most of your environment set up and runs executables in a
  different working directory. Use full paths when referencing files:
  `bazel run //path/to:target -- --read_from=/tmp/file.txt`.

#### bazel test //path/to/targets

This builds and executes one or more test targets. Bazel provides a sandboxed
testing environment and caches the results of tests. A path ending in `...` is
interpreted as a wildcard to match all targets below the specified package.
For example, to run the entire set of tests in this project, run:

``` sh
$ bazel test //...
```

Note that there are many thousands of tests in this project and running them
all may take a few hours. You probably only want to run the tests for the
particular package that you are interested in using.

#### bazel query

Whilst not strictly necessary for building and running code, `bazel query` is
a phenominally useful tool for finding your way around the codebase.

For example, you can list all of the executable targets in this project using:

```sh
$ bazel query 'kind(".*_binary",//...)'
```

Or if you are, say, working on the target `//package:foo` and what to see what
other targets will be affected by this change, run:

```sh
$ bazel query 'rdeps(//..., //package:foo)'
```

I would suggest taking a look at the
[bazel query docs](https://docs.bazel.build/versions/master/query.html)
for further details.


## 3. Known Issues

Maintaining a large project with numerous third-party dependencies is a
never-ending challenge. This section summarizes some of the key known issues
around building the code. Please check here first if you encounter an error in
the build.


#### Non-standard $PATH

The script `tools/bazel` specifies the environment in which bazel executes. It
sets only a handful of environment variables and uses common default values for
your `$PATH`. If you have installed dependencies in a non-standard path, such
as when using a python virtualenv, you may need to modify this script so that
bazel can find the right dependencies.


#### Tensorflow

The pip-installed Tensorflow dependency currently doesn't work as a result of
a bug in a dependency:
[rules_python#71](https://github.com/bazelbuild/rules_python/issues/71). To see
if you are affected, run:

```sh
$ bazel run //third_party/py/tensorflow:smoke_test
```

If the above test fails, you are affected. The workaround is to install the
version of Tensorflow specified in
`requirements.txt` into the system python packages, e.g.:

```sh
$ python3 -m pip install "tensorflow-gpu==1.14.0"
```

A symptom of this issue that you may experience breakages if you already have a
system version of Tensorflow which is incompatible with the one expected by this
project.

# Installation

## Installing the Python package

Install the ProGraML python package release using:

    pip install -U programl

ProGraML requires Python >= 3.6. The binary works on macOS and Linux (on Ubuntu
18.04, Fedora 28, Debian 10 or newer equivalents).


## Building from Source

Building from source requires a modern C++ toolchain and bazel.

### macOS

On macOS the required dependencies can be installed using
[homebrew](https://docs.brew.sh/Installation):

```sh
brew install bazelisk zlib
export LDFLAGS="-L/usr/local/opt/zlib/lib"
export CPPFLAGS="-I/usr/local/opt/zlib/include"
export PKG_CONFIG_PATH="/usr/local/opt/zlib/lib/pkgconfig"
```

Now proceed to [All platforms](#all-platforms) below.

### Linux

On debian-based linux systems, install the required toolchain using:

```sh
sudo apt install clang-9 libtinfo5 libjpeg-dev zlib1g-dev
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.7.5/bazelisk-linux-amd64 -O bazel
chmod +x bazel && mkdir -p ~/.local/bin && mv -v bazel ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"
export CC=clang
export CXX=clang++
```

### All platforms

We recommend using
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to
manage the remaining build dependencies. First create a conda environment with
the required dependencies:

    conda create -n programl python=3.8 cmake pandoc patchelf
    conda activate programl
    conda install -c conda-forge doxygen

Then clone the ProGraML source code using:

    git clone https://github.com/facebookresearch/ProGraML.git
    cd ProGraML

There are two primary git branches: `stable` tracks the latest release;
`development` is for bleeding edge features that may not yet be mature. Checkout
your preferred branch and install the python development dependencies using:

    git checkout stable
    make init

The `make init` target only needs to be run once on initial setup, or when
pulling remote changes to the ProGraML repository.

Run the test suite to confirm that everything is working:

    make test

To build and install the python package, run:

    make install

**NOTE:** To use the `programl` package that is installed by `make install` you
must leave the root directory of this repository. Attempting to import
`programl` while in the root of this repository will cause import errors.

When you are finished, you can deactivate and delete the conda environment
using:

    conda deactivate
    conda env remove -n programl


## Using ProGraML as a dependency

For Python packages you can simply add `pip freeze | grep programl >> requirements.txt`.

If you are using bazel you can add ProGraML as an external dependency. Add to
your WORKSPACE file:

```py
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name="programl",
    strip_prefix="ProGraML-<stable-commit>",
    urls=["https://github.com/ChrisCummins/ProGraML/archive/<stable-commit>.tar.gz"],
)

# ----------------- Begin ProGraML dependencies -----------------
<WORKSPACE dependencies>
# ----------------- End ProGraML dependencies -----------------
```

Where `<WORKSPACE dependencies>` is the block of delimited code in
[@programl//:WORKSPACE](https://github.com/ChrisCummins/ProGraML/blob/development/WORKSPACE)
(this is an unfortunately clumsy workaround for [recursive
workspaces](https://github.com/bazelbuild/bazel/issues/1943)).

Then in your BUILD file:

```py
cc_library(
    name = "mylib",
    srcs = ["mylib.cc"],
    deps = [
        "@programl//programl/ir/llvm",
    ],
)

py_binary(
    name = "myscript",
    srcs = ["myscript.py"],
    deps = [
        "@programl//programl/ir/llvm/py:llvm",
    ],
)
```

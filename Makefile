# This Makefile is adapted from CompilerGym's Makefile, available at:
#   https://github.com/facebookresearch/CompilerGym
#
# CompilerGym is is licensed under the MIT license.
# Copyright (c) Facebook, Inc. and its affiliates.
define HELP
ProGraML $(VERSION). Available targets:

Setting up
----------

    make init
        Install the build and runtime python dependencies. This should be run
        once before any other targets.


Testing
-------

    make test
        Run the test suite. Test results are cached so that incremental test
        runs are minimal and fast. Use this as your go-to target for testing
        modifications to the codebase.

    make itest
        Run the test suite continuously on change. This is equivalent to
        manually running `make test` when a source file is modified. Note that
        `make install-test` tests are not run. This requires bazel-watcher.
        See: https://github.com/bazelbuild/bazel-watcher#installation


Post-installation Tests
-----------------------

    make install-test
        Run the full test suite against an installed ProGraML package. This
        requires that the ProGraML package has been installed (`make
        install`). This is useful for checking the package contents but is
        usually not needed for interactive development since `make test` runs
        the same tests without having to install anything.

    make install-test-cov
        The same as `make install-test`, but with python test coverage
        reporting. A summary of test coverage is printed at the end of execution
        and the full details are recorded in a coverage.xml file in the project
        root directory.

    make install-fuzz
        Run the fuzz testing suite against an installed ProGraML package.
        Fuzz tests are tests that generate their own inputs and run in a loop
        until an error has been found, or until a minimum number of seconds have
        elapsed. This minimum time is controlled using a FUZZ_SECONDS variable.
        The default is 300 seconds (5 minutes). Override this value at the
        command line, for example `FUZZ_SECONDS=60 make install-fuzz` will run
        the fuzz tests for a minimum of one minute. This requires that the
        ProGraML package has been installed (`make install`).


Documentation
-------------

    make docs
        Build the HTML documentation using Sphinx. This is the documentation
        site that is hosted at <https://facebookresearch.github.io/ProGraML>.
        The generated HTML files are in docs/build/html.

    make livedocs
        Build the HTML documentation and serve them on localhost:8000. Changes
        to the documentation will automatically trigger incremental rebuilds
        and reload the changes.


Deployment
----------

    make bdist_wheel
        Build an optimized python wheel. The generated file is in
        dist/programl-<version>-<platform_tags>.whl

    make install
        Build and install the python wheel.

    make bdist_wheel-linux
        Use a docker container to build a python wheel for linux. This is only
        used for making release builds. This requires docker.

    bdist_wheel-linux-shell
        Drop into a bash terminal in the docker container that is used for
        linux builds. This may be useful for debugging bdist_wheel-linux
        builds.

    make bdist_wheel-linux-test
        Run the `make install-test` suite against the build artifact generated
        by `make bdist_wheel-linux`.


Tidying up
-----------

    make clean
        Remove build artifacts.

    make distclean
        Clean up all build artifacts, including the build cache.

    make uninstall
        Uninstall the python package.

    make purge
        Uninstall the python package and completely remove all datasets, logs,
        and cached files. Any experimental data or generated logs will be
        irreversibly deleted!
endef
export HELP

# Configurable paths to binaries.
CC ?= clang
CXX ?= clang++
BAZEL ?= bazel
IBAZEL ?= ibazel
PYTHON ?= python3
RSYNC ?= rsync

# Bazel build options.
BAZEL_OPTS ?=
BAZEL_BUILD_OPTS ?= -c opt
BAZEL_TEST_OPTS ?=

# The path of the repository reoot.
ROOT := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))

VERSION := $(shell cat version.txt)
OS := $(shell uname)


##############
# Setting up #
##############

.DEFAULT_GOAL := help

.PHONY: help init

help:
	@echo "$$HELP"

init:
	$(PYTHON) -m pip install -r requirements.txt
	pre-commit install


############
# Building #
############

# Files and directories generated by python disttools.
DISTTOOLS_OUTS := dist build programl.egg-info

BUILD_TARGET ?= //:py_package

bazel-build:
	$(BAZEL) $(BAZEL_OPTS) build $(BAZEL_BUILD_OPTS) $(BUILD_TARGET)

install-test-data:
	$(BAZEL) $(BAZEL_OPTS) build $(BAZEL_BUILD_OPTS) //tests/data 2>/dev/null

bdist_wheel: bazel-build
	$(PYTHON) setup.py bdist_wheel

bdist_wheel-linux-rename:
	mv dist/programl-$(VERSION)-py3-none-linux_x86_64.whl dist/programl-$(VERSION)-py3-none-manylinux2014_x86_64.whl

bdist_wheel-linux:
	rm -rf build
	docker pull chriscummins/compiler_gym-linux-build:latest
	docker run -v $(ROOT):/ProGraML --workdir /ProGraML --rm --shm-size=8g chriscummins/programl-linux-build:latest /bin/sh -c './packaging/container_init.sh && make bdist_wheel'
	mv dist/programl-$(VERSION)-py3-none-linux_x86_64.whl dist/programl-$(VERSION)-py3-none-manylinux2014_x86_64.whl
	rm -rf build

bdist_wheel-linux-shell:
	docker run -v $(ROOT):/ProGraML --workdir /ProGraML --rm --shm-size=8g -it --entrypoint "/bin/bash" chriscummins/programl-linux-build:latest

bdist_wheel-linux-test:
	docker run -v $(ROOT):/ProGraML --workdir /ProGraML --rm --shm-size=8g chriscummins/programl-linux-build:latest /bin/sh -c 'cd /ProGraML && pip3 install -U pip && pip3 install dist/programl-$(VERSION)-py3-none-manylinux2014_x86_64.whl && pip install -r tests/requirements.txt && make install-test'

all: docs bdist_wheel bdist_wheel-linux

.PHONY: bazel-build bdist_wheel bdist_wheel-linux bdist_wheel-linux-shell bdist_wheel-linux-test


###########
# Testing #
###########

programl_SITE_DATA ?= "/tmp/programl/tests/site_data"
programl_CACHE ?= "/tmp/programl/tests/cache"

# A directory that is used as the working directory for running pytest tests
# by symlinking the tests directory into it.
INSTALL_TEST_ROOT ?= "/tmp/programl/install_tests"

# The target to use. If not provided, all tests will be run. For `make test` and
# related, this is a bazel target pattern, with default value '//...'. For `make
# install-test` and related, this is a relative file path of the directory or
# file to test, with default value 'tests'.
TEST_TARGET ?=

# Extra command line arguments for pytest.
PYTEST_ARGS ?=

test:
	$(BAZEL) $(BAZEL_OPTS) test $(BAZEL_TEST_OPTS) $(if $(TEST_TARGET),$(TEST_TARGET),//...)

itest:
	$(IBAZEL) $(BAZEL_OPTS) test $(BAZEL_TEST_OPTS) $(if $(TEST_TARGET),$(TEST_TARGET),//...)

# Since we can't run ProGraML from the project root we need to jump through some
# hoops to run pytest "out of tree" by creating an empty directory and
# symlinking the test directory into it so that pytest can be invoked.
install-test-setup: install-test-data
	mkdir -p "$(INSTALL_TEST_ROOT)"
	rm -rf "$(INSTALL_TEST_ROOT)/tests" "$(INSTALL_TEST_ROOT)/tox.ini"
	mkdir "$(INSTALL_TEST_ROOT)/tests/"
	"$(RSYNC)" -a "$(ROOT)/bazel-bin/tests/data/" "$(INSTALL_TEST_ROOT)/tests/data/"
	"$(RSYNC)" -a "$(ROOT)/tests/" "$(INSTALL_TEST_ROOT)/tests/"
	ln -s "$(ROOT)/tox.ini" "$(INSTALL_TEST_ROOT)"

define pytest
	cd "$(INSTALL_TEST_ROOT)" && pytest $(if $(TEST_TARGET),$(TEST_TARGET),tests) $(1) $(PYTEST_ARGS)
endef

install-test: install-test-setup
	$(call pytest,--benchmark-disable -n auto -k "not fuzz" --durations=5)

# Note we export $CI=1 so that the tests always run as if within the CI
# environement. This is to ensure that the reported coverage matches that of
# the value on: https://codecov.io/gh/facebookresearch/ProGraML
install-test-cov: install-test-setup
	export CI=1; $(call pytest,--benchmark-disable -n auto -k "not fuzz" --durations=5 --cov=programl --cov-report=xml --cov-report=term)
	@mv "$(INSTALL_TEST_ROOT)/coverage.xml" .

# The minimum number of seconds to run the fuzz tests in a loop for. Override
# this at the commandline, e.g. `FUZZ_SECONDS=1800 make fuzz`.
FUZZ_SECONDS ?= 300

install-fuzz: install-test-setup
	$(call pytest,-p no:sugar -x -vv -k fuzz --seconds=$(FUZZ_SECONDS))

post-install-test:
	$(MAKE) -C examples/makefile_integration clean
	SEARCH_TIME=3 $(MAKE) -C examples/makefile_integration test

.PHONY: test post-install-test


################
# Installation #
################

pip-install:
	$(PYTHON) setup.py install

install: | bazel-build pip-install

.PHONY: pip-install install


##############
# Tidying up #
##############

.PHONY: clean distclean uninstall purge

clean:
	rm -rf $(DISTTOOLS_OUTS)

distclean: clean
	bazel clean --expunge

uninstall:
	$(PYTHON) -m pip uninstall -y programl
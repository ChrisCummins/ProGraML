#!/usr/bin/env bats

source labm8/sh/test.sh

teardown() {
  docker image rmi --force "bazel/tools/docker/phd_base/tests:python_version_image"
  docker image rmi --force "bazel/tools/docker/phd_base/tests:dependencies_test_image"
}

@test "run python_version" {
  docker load -i "$(DataPath phd/tools/docker/phd_base/tests/python_version_image.tar)"
  docker run --rm "bazel/tools/docker/phd_base/tests:python_version_image"
}

@test "run dependencies_test" {
  docker load -i "$(DataPath phd/tools/docker/phd_base/tests/dependencies_test_image.tar)"
  docker run --rm "bazel/tools/docker/phd_base/tests:dependencies_test_image"
}

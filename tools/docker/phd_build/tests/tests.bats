#!/usr/bin/env bats

source labm8/sh/test.sh

teardown() {
  docker image rmi --force "bazel/tools/docker/phd_build/tests:build_test_image"
}

@test "run bazel_build" {
  docker load -i "$(DataPath phd/tools/docker/phd_build/tests/build_test_image.tar)"
  docker run --rm "bazel/tools/docker/phd_build/tests:build_test_image"
}

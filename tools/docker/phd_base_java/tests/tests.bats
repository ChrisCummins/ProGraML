#!/usr/bin/env bats

source labm8/sh/test.sh

teardown() {
  docker image rmi --force "bazel/tools/docker/phd_base_java/tests:java_test_image"
}

@test "run java_test" {
  docker load -i "$(DataPath phd/tools/docker/phd_base_java/tests/java_test_image.tar)"
  docker run --rm "bazel/tools/docker/phd_base_java/tests:java_test_image"
}

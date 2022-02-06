#!/usr/bin/env bats

source labm8/sh/test.sh

setup() {
  echo "Hello, bats!" >"$BATS_TMPDIR/foo"
}

teardown() {
  # The contents of $BATS_TMPDIR is automatically cleaned up so this is not
  # necessary. It is more of a demonstration of test teardown procedure.
  rm -f "$BATS_TMPDIR/foo"
}

@test "tempdir exists" {
  test -d "$BATS_TMPDIR"
}

@test "tempdir is writable" {
  touch "$BATS_TMPDIR/a"
}

@test "setup executed" {
  test -f "$BATS_TMPDIR/foo"
}

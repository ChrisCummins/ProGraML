workspace(name="phd")

new_http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.8.0.zip",
    sha256 = "f3ed3b58511efd272eb074a3a6d6fb79d7c2e6a0e374323d1e6bcbcc1ef141bf",
    build_file = "gtest.BUILD",
    strip_prefix = "googletest-release-1.8.0/googletest",
)

new_http_archive(
    name = "benchmark",
    url = "https://github.com/google/benchmark/archive/v1.1.0.tar.gz",
    sha256 = "e7334dd254434c6668e33a54c8f839194c7c61840d52f4b6258eee28e9f3b20e",
    build_file = "benchmark.BUILD",
    strip_prefix = "benchmark-1.1.0",
)

# Intel TBB (pre-build binaries for mac and linux)

new_http_archive(
    name = "tbb_mac",
    url = "https://github.com/01org/tbb/releases/download/2017_U5/tbb2017_20170226oss_mac.tgz",
    sha256 = "6ff553ec31c33b8340ce2113853be1c42e12b1a4571f711c529f8d4fa762a1bf",
    build_file = "tbb_mac.BUILD",
    strip_prefix = 'tbb2017_20170226oss',
)

new_http_archive(
    name = "tbb_lin",
    url = "https://github.com/01org/tbb/releases/download/2017_U5/tbb2017_20170226oss_lin.tgz",
    sha256 = "c4cd712f8d58d77f7b47286c867eb6fd70a8e8aef097a5c40f6c6b53d9dd83e1",
    build_file = "tbb_lin.BUILD",
    strip_prefix = 'tbb2017_20170226oss',
)

# Experimental python rules.
# See: https://github.com/bazelbuild/rules_python

git_repository(
    name = "io_bazel_rules_python",
    remote = "https://github.com/bazelbuild/rules_python.git",
    commit = "115e3a0dab4291184fdcb0d4e564a0328364571a",
)

# Only needed for PIP support:
load("@io_bazel_rules_python//python:pip.bzl", "pip_repositories")

pip_repositories()

workspace(name = "phd")

new_http_archive(
    name = "gtest",
    build_file = "gtest.BUILD",
    sha256 = "f3ed3b58511efd272eb074a3a6d6fb79d7c2e6a0e374323d1e6bcbcc1ef141bf",
    strip_prefix = "googletest-release-1.8.0/googletest",
    url = "https://github.com/google/googletest/archive/release-1.8.0.zip",
)

new_http_archive(
    name = "benchmark",
    build_file = "benchmark.BUILD",
    sha256 = "e7334dd254434c6668e33a54c8f839194c7c61840d52f4b6258eee28e9f3b20e",
    strip_prefix = "benchmark-1.1.0",
    url = "https://github.com/google/benchmark/archive/v1.1.0.tar.gz",
)

new_http_archive(
    name = "libcxx",
    build_file = "libcxx.BUILD",
    sha256 = "70931a87bde9d358af6cb7869e7535ec6b015f7e6df64def6d2ecdd954040dd9",
    strip_prefix = "libcxx-6.0.0.src",
    url = "http://releases.llvm.org/6.0.0/libcxx-6.0.0.src.tar.xz",
)

# Intel TBB (pre-build binaries for mac and linux)

new_http_archive(
    name = "tbb_mac",
    build_file = "tbb_mac.BUILD",
    sha256 = "6ff553ec31c33b8340ce2113853be1c42e12b1a4571f711c529f8d4fa762a1bf",
    strip_prefix = "tbb2017_20170226oss",
    url = "https://github.com/01org/tbb/releases/download/2017_U5/tbb2017_20170226oss_mac.tgz",
)

new_http_archive(
    name = "tbb_lin",
    build_file = "tbb_lin.BUILD",
    sha256 = "c4cd712f8d58d77f7b47286c867eb6fd70a8e8aef097a5c40f6c6b53d9dd83e1",
    strip_prefix = "tbb2017_20170226oss",
    url = "https://github.com/01org/tbb/releases/download/2017_U5/tbb2017_20170226oss_lin.tgz",
)

# Protocol buffers.

git_repository(
    name = "org_pubref_rules_protobuf",
    remote = "https://github.com/pubref/rules_protobuf",
    tag = "v0.8.1",
)

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_repositories")

py_proto_repositories()

# Python requirements.

# TODO(cec): There is a bug in the requirements implementation which means that
# some packages do not unpack correctly. Instead of using the offical remote of
# https://github.com/bazelbuild/rules_python.git, I am using a GitHub user's
# fork which has a workaround for this.
# See: https://github.com/bazelbuild/rules_python/issues/92
git_repository(
    name = "io_bazel_rules_python",
    commit = "220c1133af2bb5c37f20c87b4c2ccfeee596ecda",
    remote = "https://github.com/jkinkead/rules_python.git",
)

load("@io_bazel_rules_python//python:pip.bzl", "pip_repositories", "pip_import")

pip_repositories()

# Add grpcio.

pip_import(
    name = "pip_grpcio",
    requirements = "@org_pubref_rules_protobuf//python:requirements.txt",
)

pip_import(
    name = "requirements",
    requirements = "//:requirements.txt",
)

load(
    "@requirements//:requirements.bzl",
    pip_grpcio_install = "pip_install",
)

pip_grpcio_install()

# LLVM, as installed by Homebrew ('system/dotfiles/run -v Llvm').

new_local_repository(
    name = "llvm_headers_mac",
    build_file = "llvm_headers.BUILD",
    path = "/usr/local/opt/llvm/include",
)

new_local_repository(
    name = "llvm_headers_linux",
    build_file = "llvm_headers.BUILD",
    path = "/home/linuxbrew/.linuxbrew/opt/llvm/include",
)

new_local_repository(
    name = "llvm_mac",
    build_file = "llvm.BUILD",
    path = "/usr/local/opt/llvm",
)

new_local_repository(
    name = "llvm_linux",
    build_file = "llvm.BUILD",
    path = "/home/linuxbrew/.linuxbrew/opt/llvm",
)

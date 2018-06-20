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

# LLVM.

new_http_archive(
    name = "llvm_mac",
    build_file = "llvm.BUILD",
    strip_prefix = "clang+llvm-6.0.0-x86_64-apple-darwin",
    url = "https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-apple-darwin.tar.xz",
)

new_http_archive(
    name = "llvm_linux",
    build_file = "llvm.BUILD",
    strip_prefix = "clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04",
    url = "https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz",
)

# Now do the same again for headers, but also strip the include/ directory.

new_http_archive(
    name = "llvm_headers_mac",
    build_file = "llvm_headers.BUILD",
    strip_prefix = "clang+llvm-6.0.0-x86_64-apple-darwin/include",
    url = "https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-apple-darwin.tar.xz",
)

new_http_archive(
    name = "llvm_headers_linux",
    build_file = "llvm_headers.BUILD",
    strip_prefix = "clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04/include",
    url = "https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz",
)

# Now do the same again for headers, but also strip the include/ directory.
# TODO: Remove these.

new_http_archive(
    name = "libcxx_mac",
    build_file = "libcxx.BUILD",
    strip_prefix = "clang+llvm-6.0.0-x86_64-apple-darwin",
    url = "https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-apple-darwin.tar.xz",
)

new_http_archive(
    name = "libcxx_linux",
    build_file = "libcxx.BUILD",
    strip_prefix = "clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04",
    url = "https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz",
)

# Intel TBB (pre-built binaries for mac and linux)

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

# Oclgrind (pre-built binaries for mac and linux).

new_http_archive(
    name = "oclgrind_mac",
    build_file = "third_party/oclgrind.BUILD",
    sha256 = "484d0d66c4bcc46526d031acb31fed52eea375e818a2b3dea3d4a31d686b3018",
    strip_prefix = "oclgrind-18.3",
    url = "https://github.com/jrprice/Oclgrind/releases/download/v18.3/Oclgrind-18.3-macOS.tgz",
)

new_http_archive(
    name = "oclgrind_linux",
    build_file = "third_party/oclgrind.BUILD",
    sha256 = "3cc8b5dfb44b948b454a9806430a7a0add915be0c1f6e2df965733ecd8b5e1fa",
    strip_prefix = "oclgrind-18.3",
    url = "https://github.com/jrprice/Oclgrind/releases/download/v18.3/Oclgrind-18.3-Linux.tgz",
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

# Bazel docker rules.
# See: https://github.com/bazelbuild/rules_docker

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "6dede2c65ce86289969b907f343a1382d33c14fbce5e30dd17bb59bb55bb6593",
    strip_prefix = "rules_docker-0.4.0",
    urls = ["https://github.com/bazelbuild/rules_docker/archive/v0.4.0.tar.gz"],
)

# Enable py3_image() rule.

load(
    "@io_bazel_rules_docker//container:container.bzl",
    "container_pull",
    container_repositories = "repositories",
)
load(
    "@io_bazel_rules_docker//python3:image.bzl",
    _py_image_repos = "repositories",
)

_py_image_repos()

# Use docker.io containers, since the Google-provided containers do not support
# specific Python versions.

container_repositories()

# TODO(cec): Use 'digest = "sha256:<sha>"' attribute instead of tag, but
# that didn't seem to work for me:
#   Error pulling and saving image
#   index.docker.io/library/python@sha256:<sha>: 'signatures'

# TODO(cec): Use the '-slim' variants of images, but that doesn't seem to work
# for me:
#   docker: Error response from daemon: OCI runtime create failed:
#   container_linux.go:348: starting container process caused
#   "exec: \"/usr/bin/python\": stat /usr/bin/python: no such file or
#   directory": unknown.

container_pull(
    name = "python3.6",
    registry = "index.docker.io",
    repository = "library/python",
    tag = "3.6",
)

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

# LLVM, as installed by Homebrew ('system/dotfiles/run -v Llvm').

new_local_repository(
    name = "llvm_mac",
    build_file_content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob(["**/*.h", "**/*.inc", "**/*.def"]),
)
""",
    path = "/usr/local/opt/llvm/include",
)

new_local_repository(
    name = "llvm_linux",
    build_file_content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob(["**/*.h", "**/*.inc", "**/*.def"]),
)
""",
    path = "/home/linuxbrew/.linuxbrew/opt/llvm/include",
)

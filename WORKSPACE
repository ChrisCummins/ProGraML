workspace(name = "programl")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

http_archive(
    name = "cec_exports_repo",
    sha256 = "6601b638e3ad93903af22a87a08044cea2f4146288a24ea3f8e1b93a6e659012",
    strip_prefix = "exports_repo-2020.06.06",
    urls = ["https://github.com/ChrisCummins/exports_repo/archive/2020.06.06.tar.gz"],
)

load("@cec_exports_repo//tools/bzl:deps.bzl", "cec_exports_repo_deps")

cec_exports_repo_deps()

http_archive(
    name = "gtest",
    sha256 = "9dc9157a9a1551ec7a7e43daea9a694a0bb5fb8bec81235d8a1e6ef64c716dcb",
    strip_prefix = "googletest-release-1.10.0",
    urls = [
        "https://github.com/google/googletest/archive/release-1.10.0.tar.gz",
    ],
)

http_archive(
    name = "com_github_google_benchmark",
    sha256 = "616f252f37d61b15037e3c2ef956905baf9c9eecfeab400cb3ad25bae714e214",
    strip_prefix = "benchmark-1.4.0",
    url = "https://github.com/google/benchmark/archive/v1.4.0.tar.gz",
)

# Google abseil C++ libraries.

# Using the current HEAD at the time of writing (2018-11-28) since the only
# release is 3 months out of date and missing some useful libraries.
http_archive(
    name = "com_google_absl",
    sha256 = "d10f684f170eb36f3ce752d2819a0be8cc703b429247d7d662ba5b4b48dd7f65",
    strip_prefix = "abseil-cpp-3088e76c597e068479e82508b1770a7ad0c806b6",
    url = "https://github.com/abseil/abseil-cpp/archive/3088e76c597e068479e82508b1770a7ad0c806b6.tar.gz",
)

# Flags library.

http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
    strip_prefix = "gflags-2.2.2",
    urls = ["https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"],
)

# Python config. Needed by pybind11_bazel.

load("//third_party/py:python_configure.bzl", "python_configure")

python_configure(name = "local_config_python")

# Pybind11.

http_archive(
    name = "pybind11",
    build_file = "//:third_party/pybind11_bazel/pybind11.BUILD",
    sha256 = "1eed57bc6863190e35637290f97a20c81cfe4d9090ac0a24f3bbf08f265eb71d",
    strip_prefix = "pybind11-2.4.3",
    urls = ["https://github.com/pybind/pybind11/archive/v2.4.3.tar.gz"],
)

# Boost C++ library.
# See: https://github.com/nelhage/rules_boost

http_archive(
    name = "com_github_nelhage_rules_boost",
    sha256 = "4031539fe0af832c6b6ed6974d820d350299a291ba7337d6c599d4854e47ed88",
    strip_prefix = "rules_boost-4ee400beca08f524e7ea3be3ca41cce34454272f",
    urls = ["https://github.com/nelhage/rules_boost/archive/4ee400beca08f524e7ea3be3ca41cce34454272f.tar.gz"],
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")

boost_deps()

# Bash testing

http_archive(
    name = "com_github_chriscummins_rules_bats",
    sha256 = "bfaa7a5818e7d6b142ac6e564f383f69f72ea593eb7de360e9aa15db69f67505",
    strip_prefix = "rules_bats-6600627545380d2b32485371bed36cef49e9ff68",
    urls = ["https://github.com/ChrisCummins/rules_bats/archive/6600627545380d2b32485371bed36cef49e9ff68.tar.gz"],
)

load("@com_github_chriscummins_rules_bats//:bats.bzl", "bats_deps")

bats_deps()

# Python config. Needed by pybind11_bazel.

load("//third_party/py:python_configure.bzl", "python_configure")

python_configure(name = "local_config_python")

# Pybind11.

http_archive(
    name = "pybind11",
    build_file = "//:third_party/pybind11_bazel/pybind11.BUILD",
    strip_prefix = "pybind11-2.4.3",
    urls = ["https://github.com/pybind/pybind11/archive/v2.4.3.tar.gz"],
)

# LLVM.

http_archive(
    name = "llvm",
    sha256 = "f90de64a5827690bfb20a5a852f42cc484fd8826f7ea7135bc9d7d31b262174f",
    strip_prefix = "bazel_llvm-20834c602493b95b4cb23214c1731776a4f24596",
    urls = ["https://github.com/ChrisCummins/bazel_llvm/archive/20834c602493b95b4cb23214c1731776a4f24596.tar.gz"],
)

load("@llvm//tools/bzl:deps.bzl", "llvm_deps")

llvm_deps()

# Intel TBB (pre-built binaries for mac and linux)

http_archive(
    name = "tbb_mac",
    build_file = "//:third_party/tbb_mac.BUILD",
    sha256 = "6ff553ec31c33b8340ce2113853be1c42e12b1a4571f711c529f8d4fa762a1bf",
    strip_prefix = "tbb2017_20170226oss",
    urls = ["https://github.com/01org/tbb/releases/download/2017_U5/tbb2017_20170226oss_mac.tgz"],
)

http_archive(
    name = "tbb_lin",
    build_file = "//:third_party/tbb_lin.BUILD",
    sha256 = "c4cd712f8d58d77f7b47286c867eb6fd70a8e8aef097a5c40f6c6b53d9dd83e1",
    strip_prefix = "tbb2017_20170226oss",
    urls = ["https://github.com/01org/tbb/releases/download/2017_U5/tbb2017_20170226oss_lin.tgz"],
)

# Protocol buffers.

http_archive(
    name = "build_stack_rules_proto",
    sha256 = "85ccc69a964a9fe3859b1190a7c8246af2a4ead037ee82247378464276d4262a",
    strip_prefix = "rules_proto-d9a123032f8436dbc34069cfc3207f2810a494ee",
    urls = ["https://github.com/stackb/rules_proto/archive/d9a123032f8436dbc34069cfc3207f2810a494ee.tar.gz"],
)

# JSON C++ library.
# https://github.com/nlohmann/json

http_archive(
    name = "nlohmann_json",
    build_file = "//:third_party/nlohmann_json.BUILD",
    sha256 = "87b5884741427220d3a33df1363ae0e8b898099fbc59f1c451113f6732891014",
    strip_prefix = "single_include",
    urls = ["https://github.com/nlohmann/json/releases/download/v3.7.3/include.zip"],
)

# pybind11 bindings for JSON.
# https://github.com/pybind/pybind11_json

http_archive(
    name = "pybind11_json",
    build_file = "//:third_party/pybind11_json.BUILD",
    sha256 = "45957f8564e921a412a6de49c578ef1faf3b04e531e859464853e26e1c734ea5",
    strip_prefix = "pybind11_json-0.2.4/include",
    urls = ["https://github.com/pybind/pybind11_json/archive/0.2.4.tar.gz"],
)

# Python requirements.

git_repository(
    name = "rules_python",
    commit = "748aa53d7701e71101dfd15d800e100f6ff8e5d1",
    remote = "https://github.com/bazelbuild/rules_python.git",
    shallow_since = "1583438240 -0500",
)

load(
    "@rules_python//python:pip.bzl",
    "pip3_import",
    "pip_repositories",
)

pip_repositories()

pip3_import(
    name = "protobuf_py_deps",
    timeout = 3600,
    requirements = "@build_stack_rules_proto//python/requirements:protobuf.txt",
)

load(
    "@protobuf_py_deps//:requirements.bzl",
    protobuf_pip_install = "pip_install",
)

protobuf_pip_install()

# Load and build all requirements.
# TODO(github.com/ChrisCummins/phd/issues/58): Break apart requirements.txt,
# using one pip3_import per package.
pip3_import(
    name = "requirements",
    timeout = 3600,
    requirements = "//:requirements.txt",
)

load(
    "@requirements//:requirements.bzl",
    pip_grpcio_install = "pip_install",
)

pip_grpcio_install()

# Python protobufs.

load("@build_stack_rules_proto//python:deps.bzl", "python_grpc_library")

python_grpc_library()

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

pip3_import(
    name = "grpc_py_deps",
    timeout = 3600,
    requirements = "@build_stack_rules_proto//python:requirements.txt",
)

load(
    "@grpc_py_deps//:requirements.bzl",
    grpc_pip_install = "pip_install",
)

grpc_pip_install()

# Needed by rules_docker.
# See: https://github.com/bazelbuild/bazel-skylib

git_repository(
    name = "bazel_skylib",
    remote = "https://github.com/bazelbuild/bazel-skylib.git",
    tag = "0.7.0",
)

# Bazel docker rules.
# See: https://github.com/bazelbuild/rules_docker

# Download the rules_docker repository at release v0.14.1
http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "dc97fccceacd4c6be14e800b2a00693d5e8d07f69ee187babfd04a80a9f8e250",
    strip_prefix = "rules_docker-0.14.1",
    urls = ["https://github.com/bazelbuild/rules_docker/releases/download/v0.14.1/rules_docker-v0.14.1.tar.gz"],
)

load(
    "@io_bazel_rules_docker//repositories:repositories.bzl",
    container_repositories = "repositories",
)

container_repositories()

load("@io_bazel_rules_docker//repositories:deps.bzl", container_deps = "deps")

container_deps()

# Enable py3_image.

load(
    "@io_bazel_rules_docker//python:image.bzl",
    _py_image_repos = "repositories",
)

_py_image_repos()

# Enable cc_image.

load(
    "@io_bazel_rules_docker//cc:image.bzl",
    _cc_image_repos = "repositories",
)

_cc_image_repos()

# My custom base images:
load(
    "@io_bazel_rules_docker//container:container.bzl",
    "container_pull",
)

# Minimal python base image.
# Defined in //tools/docker/phd_base:Dockerfile
container_pull(
    name = "phd_base",
    digest = "sha256:3fb41db45b02954e6439f5fa2fd5e0ca2ead9757575fe9125b74cf517dc13c6f",
    registry = "index.docker.io",
    repository = "chriscummins/phd_base",
)

# Same as phd_base, but with a Java environment.
# Defined in //tools/docker/phd_base_java:Dockerfile
container_pull(
    name = "phd_base_java",
    digest = "sha256:3e9c786b508e9f5471e8aeed76339e1d496a727fed80836baadb8a7a1aa69abe",
    registry = "index.docker.io",
    repository = "chriscummins/phd_base_java",
)

# Same as phd_base_java, but with Tensorflow installed.
# Defined in //tools/docker/phd_base_tf_cpu:Dockerfile
container_pull(
    name = "phd_base_tf_cpu",
    digest = "sha256:1b5ae18329edcadd7b4d0b2358b359f405117b997161b61ab94eebdb9327a8b1",
    registry = "index.docker.io",
    repository = "chriscummins/phd_base_tf_cpu",
)

# Full build environment with all required toolchains.
# Defined in //tools/docker/phd_build:Dockerfile
container_pull(
    name = "phd_build",
    digest = "sha256:47fa263c92568900f831c0cab664a97baabfafa9a3e31cff7fe058140a5ce629",
    registry = "index.docker.io",
    repository = "chriscummins/phd_build",
)

# Bazel rules for assembling and deploying software distributions.
# https://github.com/graknlabs/bazel-distribution

http_archive(
    name = "graknlabs_bazel_distribution",
    sha256 = "1bc61164a3ed85103222898c3745b64653b85c5aa05149a2cb0ad42a3e1e8a71",
    strip_prefix = "bazel-distribution-6865042c2dcfb0e877adb9ba999c9dfa855e4613",
    urls = ["https://github.com/ChrisCummins/bazel-distribution/archive/6865042c2dcfb0e877adb9ba999c9dfa855e4613.zip"],
)

pip3_import(
    name = "graknlabs_bazel_distribution_pip",
    timeout = 3600,
    requirements = "@graknlabs_bazel_distribution//pip:requirements.txt",
)

load(
    "@graknlabs_bazel_distribution_pip//:requirements.bzl",
    graknlabs_bazel_distribution_pip_install = "pip_install",
)

graknlabs_bazel_distribution_pip_install()

################################################################################
# Tensorflow sources. This is not the same as the version of tensorflow which is
# used by "//third_party/py/tensorflow". That uses the regular pip installed
# binary.
#
# This requires copying a subset of the tensorflow WORKSPACE file,
# see: https://stackoverflow.com/a/53272388

http_archive(
    name = "org_tensorflow",
    sha256 = "92116bfea188963a0e215e21e67c3494f6e1e6959f44dfbcc315f66eb70b5f83",
    strip_prefix = "tensorflow-f13f807c83c0d8d4d1ef290a17f26fe884ccfe2f",
    urls = ["https://github.com/ChrisCummins/tensorflow/archive/f13f807c83c0d8d4d1ef290a17f26fe884ccfe2f.tar.gz"],
)

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_repositories")

tf_repositories()

################################################################################

# A modern C++ formatting library.
# https://fmt.dev

http_archive(
    name = "fmt",
    build_file = "//:third_party/fmt.BUILD",
    sha256 = "1cafc80701b746085dddf41bd9193e6d35089e1c6ec1940e037fcb9c98f62365",
    strip_prefix = "fmt-6.1.2",
    urls = ["https://github.com/fmtlib/fmt/archive/6.1.2.tar.gz"],
)

# Subprocessing with modern C++.
# https://github.com/arun11299/cpp-subprocess.git

http_archive(
    name = "subprocess",
    build_file = "//:third_party/subprocess.BUILD",
    sha256 = "886df0a814a7bb7a3fdeead22f75400abd8d3235b81d05817bc8c1125eeebb8f",
    strip_prefix = "cpp-subprocess-2.0",
    urls = [
        "https://github.com/arun11299/cpp-subprocess/archive/v2.0.tar.gz",
    ],
)

http_archive(
    name = "ctpl",
    build_file = "//:third_party/ctpl.BUILD",
    sha256 = "8c1cec7c570d6d84be1d29283af5039ea27c3e69703bd446d396424bf619816e",
    strip_prefix = "CTPL-ctpl_v.0.0.2",
    urls = ["https://github.com/vit-vit/CTPL/archive/ctpl_v.0.0.2.tar.gz"],
)


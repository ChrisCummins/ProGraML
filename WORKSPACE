workspace(name = "phd")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

http_archive(
    name = "gtest",
    sha256 = "9bf1fe5182a604b4135edc1a425ae356c9ad15e9b23f9f12a02e80184c3a249c",
    strip_prefix = "googletest-release-1.8.1",
    url = "https://github.com/abseil/googletest/archive/release-1.8.1.tar.gz",
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

# Boost C++ library.
# See: https://github.com/nelhage/rules_boost

http_archive(
    name = "com_github_nelhage_rules_boost",
    sha256 = "391c6988d9f7822176fb9cf7da8930ef4474b0b35b4f24c78973cb6075fd17e4",
    strip_prefix = "rules_boost-417642961150e987bc1ac78c7814c617566ffdaa",
    url = "https://github.com/nelhage/rules_boost/archive/417642961150e987bc1ac78c7814c617566ffdaa.tar.gz",
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")

boost_deps()

# OpenCL headers.

http_archive(
    name = "opencl_120_headers",
    build_file = "//:third_party/opencl_headers.BUILD",
    sha256 = "fab4705dd3b0518f40e9d5d2f234aa57b82569841122f88a4ebcba10ecc17119",
    strip_prefix = "OpenCL-Headers-1.2/opencl12",
    urls = ["https://github.com/ChrisCummins/OpenCL-Headers/archive/v1.2.tar.gz"],
)

http_archive(
    name = "opencl_220_headers",
    build_file = "//:third_party/opencl_headers.BUILD",
    sha256 = "4b159af0ce0a5260098fff9992cde242af09c24c794ab46ff57390804a65066d",
    strip_prefix = "OpenCL-Headers-master",
    urls = ["https://github.com/ChrisCummins/OpenCL-Headers/archive/master.zip"],
)

http_archive(
    name = "libopencl",
    build_file = "//:third_party/libOpenCL.BUILD",
    sha256 = "d7c110a5ed0f26c1314f543df36e0f184783ccd11b754df396e736febbdf490a",
    strip_prefix = "OpenCL-ICD-Loader-2.2",
    urls = ["https://github.com/ChrisCummins/OpenCL-ICD-Loader/archive/v2.2.tar.gz"],
)

# LLVM.

http_archive(
    name = "llvm_mac",
    build_file = "//:third_party/llvm.BUILD",
    strip_prefix = "clang+llvm-6.0.0-x86_64-apple-darwin",
    urls = ["https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-apple-darwin.tar.xz"],
)

http_archive(
    name = "llvm_linux",
    build_file = "//:third_party/llvm.BUILD",
    strip_prefix = "clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04",
    urls = ["https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz"],
)

# Linux sources.

http_archive(
    name = "linux_srcs",
    build_file = "//:third_party/linux.BUILD",
    sha256 = "1865d2130769d4593f3c0cef4afbc9e39cdc791be218b15436a6366708142a81",
    strip_prefix = "linux-4.19",
    urls = ["https://github.com/torvalds/linux/archive/v4.19.tar.gz"],
)

# Now do the same again for headers, but also strip the include/ directory.

http_archive(
    name = "llvm_headers_mac",
    build_file = "//:third_party/llvm_headers.BUILD",
    strip_prefix = "clang+llvm-6.0.0-x86_64-apple-darwin/include",
    urls = ["https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-apple-darwin.tar.xz"],
)

http_archive(
    name = "llvm_headers_linux",
    build_file = "//:third_party/llvm_headers.BUILD",
    strip_prefix = "clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04/include",
    urls = ["https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz"],
)

http_archive(
    name = "llvm_test_suite",
    build_file = "//:third_party/llvm_test_suite.BUILD",
    strip_prefix = "test-suite-6.0.0.src",
    urls = ["http://releases.llvm.org/6.0.0/test-suite-6.0.0.src.tar.xz"],
)

# Now do the same again for headers, but also strip the include/ directory.
# TODO: Remove these.

http_archive(
    name = "libcxx_mac",
    build_file = "//:third_party/libcxx.BUILD",
    strip_prefix = "clang+llvm-6.0.0-x86_64-apple-darwin",
    urls = ["https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-apple-darwin.tar.xz"],
)

http_archive(
    name = "libcxx_linux",
    build_file = "//:third_party/libcxx.BUILD",
    strip_prefix = "clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04",
    urls = ["https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz"],
)

# Skylark rules for PlatformIO.
# See: https://github.com/mum4k/platformio_rules

git_repository(
    name = "platformio_rules",
    commit = "de06a6ff52dd38288479852ec4b8157445065fd3",
    remote = "http://github.com/ChrisCummins/platformio_rules.git",
)

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

# Oclgrind (pre-built binaries for mac and linux).

http_archive(
    name = "oclgrind_mac",
    build_file = "//:third_party/oclgrind.BUILD",
    sha256 = "484d0d66c4bcc46526d031acb31fed52eea375e818a2b3dea3d4a31d686b3018",
    strip_prefix = "oclgrind-18.3",
    urls = ["https://github.com/jrprice/Oclgrind/releases/download/v18.3/Oclgrind-18.3-macOS.tgz"],
)

http_archive(
    name = "oclgrind_linux",
    build_file = "//:third_party/oclgrind.BUILD",
    sha256 = "3cc8b5dfb44b948b454a9806430a7a0add915be0c1f6e2df965733ecd8b5e1fa",
    strip_prefix = "oclgrind-18.3",
    urls = ["https://github.com/jrprice/Oclgrind/releases/download/v18.3/Oclgrind-18.3-Linux.tgz"],
)

# CLSmith.

http_archive(
    name = "CLSmith",
    build_file = "//:third_party/CLSmith.BUILD",
    sha256 = "f37d14fdb003d60ea1dd0640efc06777428ce6debc62e470eeb05dfa128e1d07",
    strip_prefix = "CLSmith-a39a31c43c88352fc65e61dce270d8e1660cbcf0",
    urls = ["https://github.com/ChrisLidbury/CLSmith/archive/a39a31c43c88352fc65e61dce270d8e1660cbcf0.tar.gz"],
)

# bzip2.

http_archive(
    name = "bzip2",
    build_file = "//:third_party/bzip2.BUILD",
    sha256 = "ba1abd52e2798aab48f47bcc90975c0da8f6ca70dc416a0e02f02da7355710c4",
    strip_prefix = "bzip2-1.0.6",
    urls = ["https://github.com/ChrisCummins/bzip2/archive/1.0.6.tar.gz"],
)

# Data files for the Rodinia Benchmark Suite.
# Used by //datasets/benchmarks/gpgpu/rodinia-3.1.
# The code itself is checked in; this large (340 MB) archive contains the data
# files that the benchmarks use.

http_archive(
    name = "rodinia_data",
    build_file = "//:third_party/rodinia_data.BUILD",
    sha256 = "4dc9981bd1655652e9e74d8572a0c3a5876a3d818ffb0a622fc432b25f91c712",
    strip_prefix = "rodinia-3.1-data-1.0.1/data",
    urls = ["https://github.com/ChrisCummins/rodinia-3.1-data/archive/v1.0.1.tar.gz"],
)

# Protocol buffers.

http_archive(
    name = "build_stack_rules_proto",
    sha256 = "8a9cf001e3ba5c97d45ed8eb09985f15355df4bbe2dc6dd4844cccfe71f17d3e",
    strip_prefix = "rules_proto-9e68c7eb1e36bd08e9afebc094883ebc4debdb09",
    urls = ["https://github.com/stackb/rules_proto/archive/9e68c7eb1e36bd08e9afebc094883ebc4debdb09.tar.gz"],
)

# Python protobufs.

load("@build_stack_rules_proto//python:deps.bzl", "python_grpc_library")

python_grpc_library()

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load("@io_bazel_rules_python//python:pip.bzl", "pip_import", "pip_repositories")

pip_repositories()

pip_import(
    name = "protobuf_py_deps",
    requirements = "@build_stack_rules_proto//python/requirements:protobuf.txt",
)

load(
    "@protobuf_py_deps//:requirements.bzl",
    protobuf_pip_install = "pip_install",
)

protobuf_pip_install()

pip_import(
    name = "grpc_py_deps",
    requirements = "@build_stack_rules_proto//python:requirements.txt",
)

load(
    "@grpc_py_deps//:requirements.bzl",
    grpc_pip_install = "pip_install",
)

grpc_pip_install()

# Java protobufs.

load("@build_stack_rules_proto//java:deps.bzl", "java_proto_compile")

java_proto_compile()

# Java Maven dependencies.

# To add a Java Maven dependency:
#
#  1) Find a maven repo hosted online, e.g.:
#        https://mvnrepository.com/artifact/org.apache.commons/commons-jci/1.1
#  2) Find the maven repo XML description, e.g.:
#        <!-- https://mvnrepository.com/artifact/org.apache.commons/commons-jci -->
#        <dependency>
#            <groupId>org.apache.commons</groupId>
#            <artifactId>commons-jci</artifactId>
#            <version>1.1</version>
#            <type>pom</type>
#        </dependency>
#  3) Create a maven_jar() rule here.
#  4) Set the artifact property to <groupId>:<atfifactId>:<version>
#  5) Set the name to something descriptive.
#

load("//tools/bzl:maven_jar.bzl", "maven_jar")

maven_jar(
    name = "org_eclipse_jface",
    artifact = "org.eclipse.platform:org.eclipse.jface.text:3.13.0",
    sha1 = "8cbaf7f92ffc7a7daa694bd2781169b3ce7678c4",
)

maven_jar(
    name = "org_eclipse_text",
    artifact = "org.eclipse:text:3.2.0-v20060605-1400",
    attach_source = False,
    sha1 = "1b5876937d9fe612a51cd5c5023572de6fb34a42",
)

maven_jar(
    name = "org_eclipse_core_runtime",
    artifact = "org.eclipse.platform:org.eclipse.core.runtime:3.14.0",
    sha1 = "5018d6e829f976519ccf94cf4519486b2e93edfb",
)

maven_jar(
    name = "org_eclipse_swt",
    artifact = "org.eclipse.platform:org.eclipse.swt:3.107.0",
    attach_source = False,
    sha1 = "9aec4300f41685a9e84d50d7bd3715d6868c7351",
)

maven_jar(
    name = "org_eclipse_equinox_common",
    artifact = "org.eclipse.platform:org.eclipse.equinox.common:3.10.0",
    sha1 = "8758736e97bb84bf59f73013073abdc44a4e5602",
)

maven_jar(
    name = "org_eclipse_resources",
    artifact = "org.eclipse.platform:org.eclipse.core.resources:3.13.0",
    sha1 = "582ea9b37aafb39450825f82ef3a0f5867e4015c",
)

maven_jar(
    name = "org_eclipse_jobs",
    artifact = "org.eclipse.platform:org.eclipse.core.jobs:3.10.0",
    sha1 = "f7c872a52c86a304a17f6d2902c645e98cfa2dcc",
)

maven_jar(
    name = "org_eclipse_jdt_core",
    artifact = "org.eclipse.jdt:org.eclipse.jdt.core:3.14.0",
    sha1 = "16ec8157b196520fcf1ed9d7c9b63c770eda278d",
)

maven_jar(
    name = "org_eclipse_core_contenttype",
    artifact = "org.eclipse.platform:org.eclipse.core.contenttype:3.7.0",
    sha1 = "ee111d57c3fb5fd287f0dbadd3e8bb24f41ba710",
)

maven_jar(
    name = "org_eclipse_equinox_preferences",
    artifact = "org.eclipse.platform:org.eclipse.equinox.preferences:3.7.100",
    sha1 = "1d47bf96df31261867e9ed88367790aeb8d143f3",
)

maven_jar(
    name = "org_eclipse_osgi_util",
    artifact = "org.eclipse.platform:org.eclipse.osgi.util:3.5.0",
    sha1 = "2a4a95a956dde4790668d25e5993472d60456b20",
)

maven_jar(
    name = "org_eclipse_osgi",
    artifact = "org.eclipse.platform:org.eclipse.osgi:3.13.0",
    sha1 = "a71aec640e47045636f9263cc4696af7005d16e2",
)

maven_jar(
    name = "org_osgi_framework",
    artifact = "org.osgi:org.osgi.framework:1.9.0",
    sha1 = "fdc9ab6e2e7686da9329e3f23958a332cdfabfd1",
)

maven_jar(
    name = "org_osgi_service_prefs",
    artifact = "org.osgi:org.osgi.service.prefs:1.1.1",
    sha1 = "8549d2a426dce907d6b1253742ca8c11095c515b",
)

maven_jar(
    name = "com_google_protobuf_java",
    artifact = "com.google.protobuf:protobuf-java:3.6.1",
    sha1 = "0d06d46ecfd92ec6d0f3b423b4cd81cb38d8b924",
)

maven_jar(
    name = "org_eclipse_ltk_core_refactoring",
    artifact = "org.eclipse.platform:org.eclipse.ltk.core.refactoring:3.9.0",
    sha1 = "25bfb9ebf5116a67962a64ed29b5a8e29b4ab613",
)

maven_jar(
    name = "org_eclipse_ltk_ui_refactoring",
    artifact = "org.eclipse.platform:org.eclipse.ltk.ui.refactoring:3.9.100",
    sha1 = "4d1bc4ea798eaab9cca605c54f2b7efd9ace7bf7",
)

maven_jar(
    name = "com_google_guava",
    artifact = "com.google.guava:guava:23.5-jre",
    sha1 = "e9ce4989adf6092a3dab6152860e93d989e8cf88",
)

# TODO: Add sha1 checksums for maven_jars.

maven_jar(
    name = "org_apache_commons_jci_examples",
    artifact = "org.apache.commons:commons-jci-examples:1.0",
)

maven_jar(
    name = "org_apache_commons_jci_core",
    artifact = "org.apache.commons:commons-jci-core:1.0",
)

maven_jar(
    name = "org_apache_commons_jci_eclipse",
    artifact = "org.apache.commons:commons-jci-eclipse:1.0",
)

maven_jar(
    name = "org_apache_commons_jci_eclipse",
    artifact = "org.apache.commons:commons-jci-eclipse:1.0",
)

maven_jar(
    name = "org_apache_commons_cli",
    artifact = "commons-cli:commons-cli:1.0",
)

maven_jar(
    name = "org_apache_commons_logging_api",
    artifact = "commons-logging:commons-logging-api:1.1",
    attach_source = False,
)

maven_jar(
    name = "org_apache_commons_io",
    artifact = "commons-io:commons-io:1.3.1",
)

maven_jar(
    name = "asm",
    artifact = "asm:asm:2.2.1",
)

# Python requirements.

git_repository(
    name = "io_bazel_rules_python",
    commit = "6b6aedda3aab264dc1e27470655e0ae0cfb2b5bc",
    remote = "https://github.com/bazelbuild/rules_python.git",
)

load(
    "@io_bazel_rules_python//python:pip.bzl",
    "pip_import",
    "pip_repositories",
)

pip_repositories()

pip_import(
    name = "protobuf_py_deps",
    requirements = "@build_stack_rules_proto//python/requirements:protobuf.txt",
)

load(
    "@protobuf_py_deps//:requirements.bzl",
    protobuf_pip_install = "pip_install",
)

protobuf_pip_install()

pip_import(
    name = "requirements",
    requirements = "//:requirements.txt",
)

load(
    "@requirements//:requirements.bzl",
    pip_grpcio_install = "pip_install",
)

pip_grpcio_install()

# Support for standalone python binaries using par_binary().

git_repository(
    name = "subpar",
    commit = "35bb9f0092f71ea56b742a520602da9b3638a24f",
    remote = "https://github.com/google/subpar",
)

# Needed by rules_docker.
# See: https://github.com/bazelbuild/bazel-skylib

git_repository(
    name = "bazel_skylib",
    remote = "https://github.com/bazelbuild/bazel-skylib.git",
    tag = "0.7.0",
)

# Bazel docker rules.
# See: https://github.com/bazelbuild/rules_docker

http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "eda1443edcb4ef73948d9c4847f1cf046981decb075cab7331c089a78bbd368e",
    strip_prefix = "rules_docker-b79827ad2c72cfabe49dca550f69281856929c1f",
    urls = ["https://github.com/bazelbuild/rules_docker/archive/b79827ad2c72cfabe49dca550f69281856929c1f.tar.gz"],
)

# Enable py3_image() rule.

load(
    "@io_bazel_rules_docker//repositories:repositories.bzl",
    container_repositories = "repositories",
)
load("@io_bazel_rules_docker//container:container.bzl", "container_pull")
load(
    "@io_bazel_rules_docker//python3:image.bzl",
    _py_image_repos = "repositories",
)

_py_image_repos()

load(
    "@io_bazel_rules_docker//cc:image.bzl",
    _cc_image_repos = "repositories",
)

_cc_image_repos()

container_repositories()

# My custom base image for bazel-compiled binaries.
# Defined in //tools/docker/phd_base/Dockerfile.

container_pull(
    name = "base",
    digest = "sha256:0741b012da76fde785be4be3d0b1f0ac0d6d8764eeb77171ad2622ae691788af",
    registry = "index.docker.io",
    repository = "chriscummins/phd_base",
)

container_pull(
    name = "base_java",
    digest = "sha256:03c81341f676a53400abdb8000df693d764449bf8756d7b4311e781a95eb1f5b",
    registry = "index.docker.io",
    repository = "chriscummins/phd_base_java",
)

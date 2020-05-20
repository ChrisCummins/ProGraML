workspace(name = "phd")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

http_archive(
    name = "cec_exports_repo",
    sha256 = "338001b0e2e3cea978b72a456ac201b86042aecbbbdb779694ac4b131e949fc2",
    strip_prefix = "exports_repo-2020.05.06",
    urls = ["https://github.com/ChrisCummins/exports_repo/archive/2020.05.06.tar.gz"],
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

git_repository(
    name = "com_github_nelhage_rules_boost",
    commit = "4ee400beca08f524e7ea3be3ca41cce34454272f",
    remote = "https://github.com/nelhage/rules_boost",
    shallow_since = "1582750641 -0500",
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")

boost_deps()

# Bash testing

git_repository(
    name = "com_github_chriscummins_rules_bats",
    commit = "6600627545380d2b32485371bed36cef49e9ff68",
    remote = "https://github.com/ChrisCummins/rules_bats.git",
    shallow_since = "1578495032 +0000",
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
    sha256 = "0ef8e99e9c9b262a53ab8f2821e2391d041615dd3f3ff36fdf5370916b0f4268",
    strip_prefix = "clang+llvm-6.0.0-x86_64-apple-darwin",
    urls = ["https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-apple-darwin.tar.xz"],
)

http_archive(
    name = "llvm_linux",
    build_file = "//:third_party/llvm.BUILD",
    sha256 = "cc99fda45b4c740f35d0a367985a2bf55491065a501e2dd5d1ad3f97dcac89da",
    strip_prefix = "clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04",
    urls = ["https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz"],
)

# git-sizer <https://github.com/github/git-sizer/>

http_archive(
    name = "git_sizer_linux",
    build_file = "//:third_party/git-sizer.BUILD",
    sha256 = "44570533f2ba434bedb70ee90a83d65f9b4da03b008041f2dad755ba6cd47377",
    urls = ["https://github.com/github/git-sizer/releases/download/v1.3.0/git-sizer-1.3.0-linux-386.zip"],
)

http_archive(
    name = "git_sizer_mac",
    build_file = "//:third_party/git-sizer.BUILD",
    sha256 = "d80fcd2f28bfd2b531fd469bf65bd7dd2908c82a52bf4f82fdbf1caf34392124",
    urls = ["https://github.com/github/git-sizer/releases/download/v1.3.0/git-sizer-1.3.0-darwin-386.zip"],
)

# Golang and gazelle.

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "142dd33e38b563605f0d20e89d9ef9eda0fc3cb539a14be1bdb1350de2eda659",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.22.2/rules_go-v0.22.2.tar.gz",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.22.2/rules_go-v0.22.2.tar.gz",
    ],
)

http_archive(
    name = "bazel_gazelle",
    sha256 = "d8c45ee70ec39a57e7a05e5027c32b1576cc7f16d9dd37135b0eddde45cf1b10",
    urls = [
        "https://storage.googleapis.com/bazel-mirror/github.com/bazelbuild/bazel-gazelle/releases/download/v0.20.0/bazel-gazelle-v0.20.0.tar.gz",
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.20.0/bazel-gazelle-v0.20.0.tar.gz",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()

go_register_toolchains()

load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")

gazelle_dependencies()

# Linux sources.

http_archive(
    name = "linux_srcs",
    build_file = "//:third_party/linux.BUILD",
    sha256 = "1865d2130769d4593f3c0cef4afbc9e39cdc791be218b15436a6366708142a81",
    strip_prefix = "linux-4.19",
    urls = ["https://github.com/torvalds/linux/archive/v4.19.tar.gz"],
)

# Now do the same again for headers, but also strip the include/ directory.
# The reason for the duplication between @llvm_headers_XXX and @llvm_XXX is
# because the headers packages strip the include/ path prefix, so that the
# headers can be included by any cc_library with the package in the deps
# attributes, without having to set a custom -I path in copts.

http_archive(
    name = "llvm_headers_mac",
    build_file = "//:third_party/llvm_headers.BUILD",
    sha256 = "0ef8e99e9c9b262a53ab8f2821e2391d041615dd3f3ff36fdf5370916b0f4268",
    strip_prefix = "clang+llvm-6.0.0-x86_64-apple-darwin/include",
    urls = ["https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-apple-darwin.tar.xz"],
)

http_archive(
    name = "llvm_headers_linux",
    build_file = "//:third_party/llvm_headers.BUILD",
    sha256 = "cc99fda45b4c740f35d0a367985a2bf55491065a501e2dd5d1ad3f97dcac89da",
    strip_prefix = "clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04/include",
    urls = ["https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz"],
)

http_archive(
    name = "llvm_test_suite",
    build_file = "//:third_party/llvm_test_suite.BUILD",
    sha256 = "74e0055efa27b2143415148ee93b817155e6333337d9cadd4cc5d468ad3c0edf",
    strip_prefix = "test-suite-6.0.0.src",
    urls = ["http://releases.llvm.org/6.0.0/test-suite-6.0.0.src.tar.xz"],
)

# Now do the same again for headers, but also strip the include/ directory.
# TODO: Remove these.

http_archive(
    name = "libcxx_mac",
    build_file = "//:third_party/libcxx.BUILD",
    sha256 = "0ef8e99e9c9b262a53ab8f2821e2391d041615dd3f3ff36fdf5370916b0f4268",
    strip_prefix = "clang+llvm-6.0.0-x86_64-apple-darwin",
    urls = ["https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-apple-darwin.tar.xz"],
)

http_archive(
    name = "libcxx_linux",
    build_file = "//:third_party/libcxx.BUILD",
    sha256 = "cc99fda45b4c740f35d0a367985a2bf55491065a501e2dd5d1ad3f97dcac89da",
    strip_prefix = "clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04",
    urls = ["https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz"],
)

# Skylark rules for PlatformIO.
# See: https://github.com/mum4k/platformio_rules

git_repository(
    name = "platformio_rules",
    commit = "621c69fa0a890302e869d83c89fd31133e8c0e21",
    remote = "https://github.com/mum4k/platformio_rules.git",
    shallow_since = "1571659731 -0400",
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

load("//third_party/bazel:maven_jar.bzl", "maven_jar")

maven_jar(
    name = "org_junit",
    artifact = "junit:junit:4.12",
)

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
    artifact = "org.apache.commons:commons-jci-core:1.1",
)

maven_jar(
    name = "org_apache_commons_jci_eclipse",
    artifact = "org.apache.commons:commons-jci-eclipse:1.1",
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

# Support for standalone python binaries using par_binary().

git_repository(
    name = "subpar",
    commit = "35bb9f0092f71ea56b742a520602da9b3638a24f",
    remote = "https://github.com/google/subpar",
    shallow_since = "1557863961 -0400",
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

# Go dependencies.

load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies", "go_repository")

go_repository(
    name = "com_github_stretchr_testify",
    commit = "34c6fa2dc70986bccbbffcc6130f6920a924b075",
    importpath = "github.com/stretchr/testify",
)

go_repository(
    name = "com_github_jinzhu_gorm",
    commit = "a6b790ffd00da9beddc60a0d2d5b9e31f03a3ffd",
    importpath = "github.com/jinzhu/gorm",
)

# Needed by @com_github_jinzhu_gorm.
go_repository(
    name = "com_github_jinzhu_inflection",
    commit = "f5c5f50e6090ae76a29240b61ae2a90dd810112e",
    importpath = "github.com/jinzhu/inflection",
)

# Needed by @com_github_jinzhu_gorm.
go_repository(
    name = "com_github_mattn_go_sqlite3",
    commit = "5dd71670cca4bc0ee90371eabd0f1bdba1ac6f35",
    importpath = "github.com/mattn/go-sqlite3",
)

go_repository(
    name = "com_github_bazelbuild_buildtools",
    commit = "c4e649df7ade24c3662729fdabd7bcff67866fef",
    importpath = "github.com/bazelbuild/buildtools",
)

go_repository(
    name = "com_github_golang_go",
    commit = "641e61db57f176e33828ed5354810fa3f13ac76d",
    importpath = "github.com/golang/gp",
)

# Pre-built go binaries.

http_archive(
    name = "go_linux",
    build_file = "//:third_party/go.BUILD",
    sha256 = "a1bc06deb070155c4f67c579f896a45eeda5a8fa54f35ba233304074c4abbbbd",
    strip_prefix = "go",
    urls = ["https://dl.google.com/go/go1.13.6.linux-amd64.tar.gz"],
)

http_archive(
    name = "go_mac",
    build_file = "//:third_party/go.BUILD",
    sha256 = "1ee0dc6a7abf389dac898cbe27e28c4388a61e45cba2632c01d749e25003007f",
    strip_prefix = "go",
    urls = ["https://dl.google.com/go/go1.13.6.darwin-amd64.tar.gz"],
)

# Pre-compiled shfmt binary
# https://github.com/mvdan/sh

http_file(
    name = "shfmt_linux",
    downloaded_file_path = "shfmt",
    executable = 1,
    sha256 = "86892020280d923976ecaaad1e7db372d37dce3cfaad44a7de986f7eb728eae7",
    urls = ["https://github.com/mvdan/sh/releases/download/v3.0.1/shfmt_v3.0.1_linux_amd64"],
)

http_file(
    name = "shfmt_mac",
    downloaded_file_path = "shfmt",
    executable = 1,
    sha256 = "e470d216818a107078fbaf34807079c4857cb98610d67c96bf4dece43a56b66c",
    urls = ["https://github.com/mvdan/sh/releases/download/v3.0.1/shfmt_v3.0.1_darwin_amd64"],
)

# Pre-compiled buf binary
# https://buf.build/

http_file(
    name = "buf_linux",
    downloaded_file_path = "buf",
    executable = 1,
    sha256 = "3d349baa67ce7ff063074e8aff5fb19e90143ca4cdadaa095c2cac1e55afac92",
    urls = ["https://github.com/bufbuild/buf/releases/download/v0.5.0/buf-Linux-x86_64"],
)

http_file(
    name = "buf_darwin",
    downloaded_file_path = "buf",
    executable = 1,
    sha256 = "6da8b67e68b3e386f49bc36d2ca785d766e1420366e5c270a204ea81a76197c7",
    urls = ["https://github.com/bufbuild/buf/releases/download/v0.5.0/buf-Darwin-x86_64"],
)

# Pre-compiled prototool binary
# https://github.com/uber/prototool

http_file(
    name = "prototool_linux",
    downloaded_file_path = "prototool",
    executable = 1,
    sha256 = "e1e3d81228f7d157d9476034dd1b21005926c96dcd8b7d220074ac071304545e",
    urls = ["https://github.com/uber/prototool/releases/download/v1.9.0/prototool-Linux-x86_64"],
)

http_file(
    name = "prototool_darwin",
    downloaded_file_path = "prototool",
    executable = 1,
    sha256 = "8cb668edbd51a13f057535d8ca7e4b0574fe7ae092338714b3d4b513cf188220",
    urls = ["https://github.com/uber/prototool/releases/download/v1.9.0/prototool-Darwin-x86_64"],
)

# My fork of black fixed for 2 spaces.

http_archive(
    name = "black2-spaces",
    build_file = "//third_party/py:black.BUILD",
    sha256 = "fce1caed01aca58f3239f88bdd9045bf950e9de3b8a2a5b5ed973037d819a6a4",
    strip_prefix = "black-2spaces-8f1fe951bcade3792f586eb52d45e38a6428c0fd",
    urls = ["https://github.com/ChrisCummins/black-2spaces/archive/8f1fe951bcade3792f586eb52d45e38a6428c0fd.zip"],
)

################################################################################
# Tensorflow sources. This is not the same as the version of tensorflow which is
# used by "//third_party/py/tensorflow". That uses the regular pip installed
# binary.
#
# This requires copying a subset of the tensorflow WORKSPACE file,
# see: https://stackoverflow.com/a/53272388

git_repository(
    name = "org_tensorflow",
    commit = "f13f807c83c0d8d4d1ef290a17f26fe884ccfe2f",
    remote = "https://github.com/ChrisCummins/tensorflow.git",
    shallow_since = "1581275770 +0000",
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

# Build rules for interfacing with "foreign" (non-Bazel) build systems.
# https://github.com/bazelbuild/rules_foreign_cc

http_archive(
    name = "rules_foreign_cc",
    sha256 = "0a2a0099c837ddfc365eca74b54f33790fa2542e4fdfbf50e4ad0f22d4821917",
    strip_prefix = "rules_foreign_cc-8372f383cf7277a88762efe25d8cfee10ad27929",
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/8372f383cf7277a88762efe25d8cfee10ad27929.zip",
)

all_content = """filegroup(name = "all", srcs = glob(["**"]), visibility = ["//visibility:public"])"""

load("@rules_foreign_cc//:workspace_definitions.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies([])

# The C++ connector for PostgreSQL.
# https://www.postgresql.org/docs/6.5/libpqplusplus.htm

http_archive(
    name = "libpqxx",
    build_file_content = all_content,
    sha256 = "2280621eee7ec675ed6751abac2273f0ac41395e99be96e06584a54a0cc3985a",
    strip_prefix = "libpqxx-6.4.6",
    urls = ["https://github.com/jtv/libpqxx/archive/6.4.6.tar.gz"],
)

# A flexible ASCII progress-bar for C++.
# https://github.com/prakhar1989/progress-cpp

http_archive(
    name = "progress_cpp",
    build_file = "//:third_party/progress_cpp.BUILD",
    sha256 = "e5a8804c71a4b86149a00deab62bede041003df7109822805ea8cce015d5373b",
    strip_prefix = "progress-cpp-06fdda086eb06b462faff7a917984062bc15b61e",
    urls = ["https://github.com/prakhar1989/progress-cpp/archive/06fdda086eb06b462faff7a917984062bc15b61e.tar.gz"],
)

# OpenSSL.

http_archive(
    name = "openssl",
    build_file_content = all_content,
    sha256 = "23011a5cc78e53d0dc98dfa608c51e72bcd350aa57df74c5d5574ba4ffb62e74",
    strip_prefix = "openssl-OpenSSL_1_1_1d",
    urls = ["https://github.com/openssl/openssl/archive/OpenSSL_1_1_1d.tar.gz"],
)

# MNIST dataset

http_file(
    name = "mnist_train_images",
    sha256 = "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    urls = ["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"],
)

http_file(
    name = "mnist_train_labels",
    sha256 = "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    urls = ["http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"],
)

http_file(
    name = "mnist_test_images",
    sha256 = "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    urls = ["http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"],
)

http_file(
    name = "mnist_test_labels",
    sha256 = "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
    urls = ["http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"],
)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")


def programl_deps():
  http_archive(
    name="labm8",
    sha256="e4bc669322533e7615f689e5e8a810446d0c803be2e3b21e99a4e0135709755e",
    strip_prefix="labm8-b98301dec615465a6567bed4ec4131753d1f8b32",
    urls=[
      "https://github.com/ChrisCummins/labm8/archive/b98301dec615465a6567bed4ec4131753d1f8b32.tar.gz"
    ],
  )

  http_archive(
    name="com_github_nelhage_rules_boost",
    sha256="4031539fe0af832c6b6ed6974d820d350299a291ba7337d6c599d4854e47ed88",
    strip_prefix="rules_boost-4ee400beca08f524e7ea3be3ca41cce34454272f",
    urls=[
      "https://github.com/nelhage/rules_boost/archive/4ee400beca08f524e7ea3be3ca41cce34454272f.tar.gz"
    ],
  )

  http_archive(
    name="llvm",
    sha256="ea0dbab56d11e78006c68c39bc99da672bb6adc7ca03237ba4eb88887bf91a93",
    strip_prefix="bazel_llvm-ae9ef2a711c5744fe52c5666d76976a3c6a3128b",
    urls=[
      "https://github.com/ChrisCummins/bazel_llvm/archive/ae9ef2a711c5744fe52c5666d76976a3c6a3128b.tar.gz"
    ],
  )

  http_archive(
    name="rules_python",
    sha256="64a3c26f95db470c32ad86c924b23a821cd16c3879eed732a7841779a32a60f8",
    strip_prefix="rules_python-748aa53d7701e71101dfd15d800e100f6ff8e5d1",
    urls=[
      "https://github.com/bazelbuild/rules_python/archive/748aa53d7701e71101dfd15d800e100f6ff8e5d1.tar.gz"
    ],
  )

  http_archive(
    name="com_github_chriscummins_rules_bats",
    strip_prefix="rules_bats-6600627545380d2b32485371bed36cef49e9ff68",
    sha256="bfaa7a5818e7d6b142ac6e564f383f69f72ea593eb7de360e9aa15db69f67505",
    urls=[
      "https://github.com/ChrisCummins/rules_bats/archive/6600627545380d2b32485371bed36cef49e9ff68.tar.gz"
    ],
  )

  http_archive(
    name="subprocess",
    build_file="@programl//:third_party/subprocess.BUILD",
    sha256="886df0a814a7bb7a3fdeead22f75400abd8d3235b81d05817bc8c1125eeebb8f",
    strip_prefix="cpp-subprocess-2.0",
    urls=["https://github.com/arun11299/cpp-subprocess/archive/v2.0.tar.gz",],
  )

  http_archive(
    name="ctpl",
    build_file="@programl//:third_party/ctpl.BUILD",
    sha256="8c1cec7c570d6d84be1d29283af5039ea27c3e69703bd446d396424bf619816e",
    strip_prefix="CTPL-ctpl_v.0.0.2",
    urls=["https://github.com/vit-vit/CTPL/archive/ctpl_v.0.0.2.tar.gz"],
  )

  http_archive(
    name="fmt",
    build_file="@programl//:third_party/fmt.BUILD",
    sha256="1cafc80701b746085dddf41bd9193e6d35089e1c6ec1940e037fcb9c98f62365",
    strip_prefix="fmt-6.1.2",
    urls=["https://github.com/fmtlib/fmt/archive/6.1.2.tar.gz"],
  )

  http_archive(
    name="pybind11_json",
    build_file="@programl//:third_party/pybind11_json.BUILD",
    sha256="45957f8564e921a412a6de49c578ef1faf3b04e531e859464853e26e1c734ea5",
    strip_prefix="pybind11_json-0.2.4/include",
    urls=["https://github.com/pybind/pybind11_json/archive/0.2.4.tar.gz"],
  )

  http_archive(
    name="nlohmann_json",
    build_file="@programl//:third_party/nlohmann_json.BUILD",
    sha256="87b5884741427220d3a33df1363ae0e8b898099fbc59f1c451113f6732891014",
    strip_prefix="single_include",
    urls=[
      "https://github.com/nlohmann/json/releases/download/v3.7.3/include.zip"
    ],
  )

  http_archive(
    name="build_stack_rules_proto",
    sha256 = "d456a22a6a8d577499440e8408fc64396486291b570963f7b157f775be11823e",
    strip_prefix="rules_proto-b2913e6340bcbffb46793045ecac928dcf1b34a5",
    urls=[
      "https://github.com/stackb/rules_proto/archive/b2913e6340bcbffb46793045ecac928dcf1b34a5.tar.gz"
    ],
  )

  http_archive(
    name="tbb_mac",
    build_file="@programl//:third_party/tbb_mac.BUILD",
    sha256="6ff553ec31c33b8340ce2113853be1c42e12b1a4571f711c529f8d4fa762a1bf",
    strip_prefix="tbb2017_20170226oss",
    urls=[
      "https://github.com/01org/tbb/releases/download/2017_U5/tbb2017_20170226oss_mac.tgz"
    ],
  )

  http_archive(
    name="tbb_lin",
    build_file="@programl//:third_party/tbb_lin.BUILD",
    sha256="c4cd712f8d58d77f7b47286c867eb6fd70a8e8aef097a5c40f6c6b53d9dd83e1",
    strip_prefix="tbb2017_20170226oss",
    urls=[
      "https://github.com/01org/tbb/releases/download/2017_U5/tbb2017_20170226oss_lin.tgz"
    ],
  )

  http_archive(
    name="pybind11",
    build_file="@programl//:third_party/pybind11_bazel/pybind11.BUILD",
    sha256="1eed57bc6863190e35637290f97a20c81cfe4d9090ac0a24f3bbf08f265eb71d",
    strip_prefix="pybind11-2.4.3",
    urls=["https://github.com/pybind/pybind11/archive/v2.4.3.tar.gz"],
  )

  http_archive(
    name="com_google_absl",
    sha256="d10f684f170eb36f3ce752d2819a0be8cc703b429247d7d662ba5b4b48dd7f65",
    strip_prefix="abseil-cpp-3088e76c597e068479e82508b1770a7ad0c806b6",
    url="https://github.com/abseil/abseil-cpp/archive/3088e76c597e068479e82508b1770a7ad0c806b6.tar.gz",
  )

  http_archive(
    name="com_github_gflags_gflags",
    sha256="34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
    strip_prefix="gflags-2.2.2",
    urls=["https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"],
  )

  http_archive(
    name="gtest",
    sha256="9dc9157a9a1551ec7a7e43daea9a694a0bb5fb8bec81235d8a1e6ef64c716dcb",
    strip_prefix="googletest-release-1.10.0",
    urls=[
      "https://github.com/google/googletest/archive/release-1.10.0.tar.gz",
    ],
  )

  http_archive(
    name="com_github_google_benchmark",
    sha256="616f252f37d61b15037e3c2ef956905baf9c9eecfeab400cb3ad25bae714e214",
    strip_prefix="benchmark-1.4.0",
    url="https://github.com/google/benchmark/archive/v1.4.0.tar.gz",
  )

  http_archive(
    name="org_tensorflow",
    sha256="92116bfea188963a0e215e21e67c3494f6e1e6959f44dfbcc315f66eb70b5f83",
    strip_prefix="tensorflow-f13f807c83c0d8d4d1ef290a17f26fe884ccfe2f",
    urls=[
      "https://github.com/ChrisCummins/tensorflow/archive/f13f807c83c0d8d4d1ef290a17f26fe884ccfe2f.tar.gz"
    ],
  )

  http_archive(
    name="io_bazel_rules_closure",
    sha256="5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix="rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls=[
      "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
      "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
  )

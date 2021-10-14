load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")


def programl_deps():
  http_archive(
    name="labm8",
    sha256 = "d31d9e6850552967bae128dc466a378a0476bfd3c53ab05533ac1206e8f43b3e",
    strip_prefix="labm8-913d67f4f454cedc61220a210113bbf0460bb4d5",
    urls=[
      "https://github.com/ChrisCummins/labm8/archive/913d67f4f454cedc61220a210113bbf0460bb4d5.tar.gz"
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
    sha256 = "646f0da3a57909d8e479253a8a9274b866d28307e31b23fff8475aefcc8157b9",
    strip_prefix="bazel_llvm-cb4efd451e3c71b14663b38cfbe3348d3cbd485b",
    urls=[
      "https://github.com/ChrisCummins/bazel_llvm/archive/cb4efd451e3c71b14663b38cfbe3348d3cbd485b.tar.gz"
    ],
  )

  http_archive(
      name = "rules_python",
      sha256 = "b5668cde8bb6e3515057ef465a35ad712214962f0b3a314e551204266c7be90c",
      strip_prefix = "rules_python-0.0.2",
      urls = [
          "https://github.com/bazelbuild/rules_python/releases/download/0.0.2/rules_python-0.0.2.tar.gz",
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
    name="nlohmann_json",
    build_file="@programl//:third_party/nlohmann_json.BUILD",
    sha256="87b5884741427220d3a33df1363ae0e8b898099fbc59f1c451113f6732891014",
    strip_prefix="single_include",
    urls=[
      "https://github.com/nlohmann/json/releases/download/v3.7.3/include.zip"
    ],
  )

  http_archive(
      name = "rules_proto",
      sha256 = "8e7d59a5b12b233be5652e3d29f42fba01c7cbab09f6b3a8d0a57ed6d1e9a0da",
      strip_prefix = "rules_proto-7e4afce6fe62dbff0a4a03450143146f9f2d7488",
      urls = [
          "https://mirror.bazel.build/github.com/bazelbuild/rules_proto/archive/7e4afce6fe62dbff0a4a03450143146f9f2d7488.tar.gz",
          "https://github.com/bazelbuild/rules_proto/archive/7e4afce6fe62dbff0a4a03450143146f9f2d7488.tar.gz",
      ],
  )

  http_archive(
      name = "com_github_grpc_grpc",
      sha256 = "7e287b23cede28770bba58581a1d5683fd619fb748c4c2df8fcc3363957b1127",
      strip_prefix = "grpc-014a2a0d54849fff795efbcb96b18416cbd7e189",
      urls = [
          "https://github.com/grpc/grpc/archive/014a2a0d54849fff795efbcb96b18416cbd7e189.tar.gz",
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
    name="com_google_absl",
    sha256="35f22ef5cb286f09954b7cc4c85b5a3f6221c9d4df6b8c4a1e9d399555b366ee",
    strip_prefix="abseil-cpp-997aaf3a28308eba1b9156aa35ab7bca9688e9f6",
    urls=[
        "https://storage.googleapis.com/grpc-bazel-mirror/github.com/abseil/abseil-cpp/archive/997aaf3a28308eba1b9156aa35ab7bca9688e9f6.tar.gz",
        "https://github.com/abseil/abseil-cpp/archive/997aaf3a28308eba1b9156aa35ab7bca9688e9f6.tar.gz",
    ],
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

workspace(name = "programl")

# === Begin ProGraML dependencies ===
load("@programl//tools:bzl/deps.bzl", "programl_deps")

programl_deps()

# Boost.
load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")

boost_deps()

# Python config. Needed by pybind11_bazel.
load("@programl//third_party/py:python_configure.bzl", programl_python_configure = "python_configure")

programl_python_configure(name = "local_config_python")

# LLVM.
load("@llvm//tools/bzl:deps.bzl", "llvm_deps")

llvm_deps()

# Bats.
load("@com_github_chriscummins_rules_bats//:bats.bzl", "bats_deps")

bats_deps()

# Python requirements.
load(
    "@rules_python//python:pip.bzl",
    "pip3_import",
    "pip_repositories",
)

pip_repositories()

# ProGraML pip requirements.

pip3_import(
    name = "programl_requirements",
    timeout = 3600,
    requirements = "@programl//:requirements.txt",
)

load(
    "@programl_requirements//:requirements.bzl",
    programl_pip_install = "pip_install",
)

programl_pip_install()

# TensorFlow pip requirements.

pip3_import(
    name = "programl_tensorflow_requirements",
    timeout = 3600,
    requirements = "@programl//third_party/py/tensorflow:requirements.txt",
)

load(
    "@programl_tensorflow_requirements//:requirements.bzl",
    programl_pip_install = "pip_install",
)

programl_pip_install()

# Protobuf.
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

# Tensorflow.
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_repositories")

tf_repositories()
# === End ProGraML dependencies ===

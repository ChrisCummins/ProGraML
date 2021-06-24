workspace(name = "programl")

# ----------------- Begin ProGraML dependencies -----------------
load("@programl//tools:bzl/deps.bzl", "programl_deps")

programl_deps()

# === Boost ===

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")

boost_deps()

# === LLVM ===

load("@llvm//tools/bzl:deps.bzl", "llvm_deps")

llvm_deps()

# === Bats ===

load("@com_github_chriscummins_rules_bats//:bats.bzl", "bats_deps")

bats_deps()

# === Protocol buffers ===

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")

rules_proto_dependencies()

rules_proto_toolchains()

# === GRPC ===

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

grpc_extra_deps()

# ----------------- End ProGraML dependencies -----------------

# Top level package of the phd repo.

load("//tools/bzl:string_genrule.bzl", "python_string_genrule")
load("@bazel_gazelle//:def.bzl", "gazelle")
load("@build_stack_rules_proto//python:python_grpc_library.bzl", "python_grpc_library")

exports_files([
    "README.md",
    "WORKSPACE",
    "version.txt",
    "deployment.properties",
])

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

# Golang.
# Gazelle directive:
# gazelle:prefix github.com/ChrisCummins/phd
# gazelle:proto disable

gazelle(name = "gazelle")

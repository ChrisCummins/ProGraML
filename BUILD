# Top-level package of the workspace.

load("@bazel_gazelle//:def.bzl", "gazelle")

exports_files([
    "README.md",
    "INSTALL.md",
    "CONTRIBUTING.md",
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

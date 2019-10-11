# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache.

py_library(
    name = "fs",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("fs"),
        requirement("fs.sshfs"),
        requirement("paramiko"),
        requirement("appdirs"),
        requirement("property-cached"),
    ],
)

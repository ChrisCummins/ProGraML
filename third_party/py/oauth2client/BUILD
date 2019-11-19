# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache v2

py_library(
    name = "oauth2client",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("oauth2client"),
        requirement("httplib2"),
        requirement("rsa"),
        requirement("pyasn1"),
        requirement("pyasn1_modules"),
    ],
)

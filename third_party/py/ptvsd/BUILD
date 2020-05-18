# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

# NOTE: In testing I've found that the ptvsd dependency can fail to pull in
# the ptvsd module. This is probably related to issue:
# github.com/bazelbuild/rules_python/issues/71
# If your dependency on //third_party/py/ptvsd fails, install it globally using:
#    $ /usr/local/bin/python3 -m pip install ptvsd
py_library(
    name = "ptvsd",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("ptvsd"),
    ],
)

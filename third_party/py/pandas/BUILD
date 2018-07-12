# A wrapper around pandas pip package to pull in undeclared dependencies.
#
# If a python target requires pandas, it should depend on this package
# (i.e. //third_party/py/pandas), instead of requirement("pandas").

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"]) # BSD

py_library(
    name = "pandas",
    srcs = ["pandas.py"],
    deps = [
        requirement("pandas"),
        requirement("numpy"),
        requirement("pytz"),
        requirement("python-dateutil"),
  ],
)

# A wrapper around pandas pip package to pull in undeclared dependencies.
#
# If a python target requires pandas, it should depend on this package
# (i.e. //third_party/py/pandas), instead of requirement("pandas").

load("@programl_requirements//:requirements.bzl", "requirement")
load("@tasks_requirements//:requirements.bzl", task_requirement = "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

# Requires pandas!
py_library(
    name = "pandas",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        task_requirement("pandas"),
        task_requirement("python-dateutil"),
        task_requirement("pytz"),
        requirement("six"),
        requirement("numpy"),
    ],
)

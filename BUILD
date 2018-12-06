# Top level package of the phd repo.

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

filegroup(
    name = "config",
    srcs = ["config.pbtxt"],
    visibility = ["//config:__subpackages__"],
)

filegroup(
    name = "configure_py",
    srcs = ["configure"],
)

py_test(
    name = "configure_test",
    srcs = ["configure_test.py"],
    data = [":configure_py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//labm8:bazelutil",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

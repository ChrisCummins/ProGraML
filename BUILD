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

py_library(
    name = "conftest",
    testonly = 1,
    srcs = ["conftest.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8:app",
        "//third_party/py/pytest",
    ],
)

py_test(
    name = "configure_test",
    srcs = ["configure_test.py"],
    data = [":configure_py"],
    deps = [
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:test",
    ],
)

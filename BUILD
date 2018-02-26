# clgen sources and data glob.
sh_library(
    name = "labm8",
    srcs = glob([
        "labm8/*.py",
        "make/**/*",
        "Makefile",
        "requirements.txt",
        "setup.py",
        "setup.cfg",
        "tests/**/*",
    ]),
    visibility = ["//visibility:public"],
)

# a script which sets up a virtualenv and runs the test suite.
sh_test(
    name = "main",
    timeout = "eternal",
    srcs = ["tests/.runner.sh"],
    args = [
        "src/labm8",
        "python3.6",
    ],
    deps = [":labm8"],
)

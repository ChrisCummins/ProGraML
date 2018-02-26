cc_library(
    name = 'opencl',
    hdrs = [
        'third_party/opencl/include/cl.h',
        'third_party/opencl/include/cl.hpp'
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

# The authoritative repo python interpreter. Install using:
#   $ virtualenv -p python3.6 venv/phd
#   $ source venv/phd/bin/activate
#   $ pip uninstall -y setuptools
#
# The last command is an annoying workaround because setuptools contains
# filenames with spaces in them, which bazel does not support.
py_runtime(
    name = "python3.6",
    files = glob(["venv/phd/**"]),
    interpreter = "venv/phd/bin/python3.6",
    visibility = ["//visibility:public"],
)

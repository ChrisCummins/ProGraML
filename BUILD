cc_library(
    name = "opencl",
    hdrs = [
        "third_party/opencl/include/cl.h",
        "third_party/opencl/include/cl.hpp",
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
py_runtime(
    name = "python3.6",
    files = glob(
        ["venv/phd/**"],
        exclude = [
            # Illegal as Bazel labels but are not required by pip.
            "**/launcher manifest.xml",
            "**/setuptools/*.tmpl",
        ],
    ),
    interpreter = "venv/phd/bin/python3.6",
    visibility = ["//visibility:public"],
)

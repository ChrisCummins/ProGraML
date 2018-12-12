# Package containing Jupyter notebook servers.

# A Jupyter notebook server that includes a bunch of dependencies which I
# find useful. If you would like to include extra dependencies, create a new
# py_binary target with this target as the srcs, and add the dependencies:
#
#    py_binary(
#      name = "notebook",
#      srcs = ["//notebooks"],
#      deps = [
#          # Your dependencies here.
#      ],
#    )
#
# Use the run_notebooks.sh script to launch this target in the root of the
# source tree: bash notebooks/run_notebooks.sh.
py_binary(
    name = "notebooks",
    srcs = ["notebooks.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8:bazelutil",
        "//labm8:labdate",
        "//labm8:labtypes",
        "//labm8:pbutil",
        "//labm8:sqlutil",
        "//labm8:viz",
        "//third_party/py/absl",
        "//third_party/py/bokeh",
        "//third_party/py/graph_nets",
        "//third_party/py/jupyter",
        "//third_party/py/keras",
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/seaborn",
        "//third_party/py/sonnet",
        "//third_party/py/tensorflow",
    ],
)

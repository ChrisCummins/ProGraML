# Support for OpenMP.

# A dummy target for compiling with OpenMP features enabled.
#
# This "library" contains no flags, but defines the compile-time
# and link-time options for compiling programs which use OpenMP
# pragmas.
#
# Add this target to your C++ binary/libraryy deps to enable
# OpenMP, and add the "-fopenmp" copt. For example:
#
#   cc_binary(
#       name = "my_app",
#       srcs = ["my_app.cc"],
#       copts = ["-fopenmp"],
#       deps = ["//third_party/openmp"],
#   )
#
cc_library(
    name = "openmp",
    copts = ["-fopenmp"],
    linkopts = [
        "-fopenmp",
    ] + select({
        "//:darwin": [
            "-L/usr/local/Cellar/libomp/10.0.0/lib",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

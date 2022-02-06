# bzip2 data compressor.
# See: http://www.bzip.org/

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "bzlib_srcs",
    srcs = [
        "blocksort.c",
        "bzlib.c",
        "bzlib.h",
        "bzlib_private.h",
        "compress.c",
        "crctable.c",
        "decompress.c",
        "huffman.c",
        "randtable.c",
    ],
)

filegroup(
    name = "bzip2_srcs",
    srcs = [
        "bzip2.c",
        ":bzlib_srcs",
    ],
)

cc_binary(
    name = "bzip2",
    srcs = [
        ":bzip2_srcs",
    ],
    deps = [
        ":bzlib",
    ],
)

cc_library(
    name = "bzlib",
    srcs = [
        ":bzlib_srcs",
    ],
)

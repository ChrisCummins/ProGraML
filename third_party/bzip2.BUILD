# bzip2 data compressor.
# See: http://www.bzip.org/

cc_binary(
  name = "bzip2",
  srcs = [
    'bzip2.c',
  ],
  deps = [
    ":bzlib",
  ]
)

cc_library(
  name = "bzlib",
  srcs = glob([
    'blocksort.c',
    'bzlib.c',
    'bzlib.h',
    'bzlib_private.h',
    'compress.c',
    'crctable.c',
    'decompress.c',
    'huffman.c',
    'randtable.c',
  ])
)

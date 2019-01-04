# Data from published results.

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "embeddings",
    srcs = ["emb.p"],
)

# Data for devmap task, downloaded from:
# https://polybox.ethz.ch/index.php/s/U08Z3xLhvbLk8io/download
filegroup(
    name = "task_devmap",
    srcs = ["task_devmap.zip"],
)

# The above zipfile, but only the 'kernels_cl' dir, and with missing
# npb-3.3-SP-initialize*.cl files added.
filegroup(
    name = "task_devmap_kernels",
    srcs = ["task_devmap_kernels.zip"],
)

# Data for threadcoarsening task, downloaded from:
# https://polybox.ethz.ch/index.php/s/Dl8v8dKbuoWS3Ck/download
filegroup(
    name = "task_threadcoarsening",
    srcs = ["task_threadcoarsening.zip"],
)

filegroup(
    name = "vocabulary",
    srcs = ["vocabulary.zip"],
)

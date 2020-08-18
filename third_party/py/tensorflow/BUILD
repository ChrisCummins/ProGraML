load("@programl_tensorflow_requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0.

py_library(
    name = "tensorflow",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("tensorflow"),
        # Copied from:
        # https://github.com/tensorflow/tensorflow/blob/f3a015274fadab00ec8cad92af2a968e0ecd434f/tensorflow/tools/pip_package/setup.py#L54-L73
        requirement("absl-py"),
        requirement("astunparse"),
        requirement("flatbuffers"),
        requirement("gast"),
        requirement("google_pasta"),
        requirement("h5py"),
        requirement("keras_preprocessing"),
        requirement("numpy"),
        requirement("opt_einsum"),
        requirement("protobuf"),
        requirement("six"),
        requirement("tensorboard"),
        requirement("tensorflow_estimator"),
        requirement("termcolor"),
        requirement("wheel"),
        requirement("wrapt"),
        "//third/party/py/keras_preprocessing",
    ],
)

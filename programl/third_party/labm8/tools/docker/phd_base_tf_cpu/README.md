# Docker image: phd_base_tf_cpu

An extension of `phd_base_java` with CPU-only Tensorflow.

## Updating [chriscummins/phd_base_tf_cpu](https://hub.docker.com/r/chriscummins/phd_base_tf_cpu)

1. Set a new project version in `//:version.txt`.
2. Modify `tools/docker/phd_base_tf_cpu/Dockerfile` with your desired changes.
3. Build and export the image using: `$ bazel run //tools/docker:export -- $PHD/tools/docker/phd_base_tf_cpu`.
4. Update the digest in the `WORKSPACE` file:
```
container_pull(
    name = "phd_base_tf_cpu",
    digest = "sha256:<new_digest>",
    registry = "index.docker.io",
    repository = "chriscummins/phd_base_tf_cpu",
)
```
5. Run the test suite: `$ bazel test //tools/docker/phd_build/...`.

# Docker image: phd

A self-contained environment configured to build and run the code in this 
project.

Build this image using:

```sh
$ docker build -t phd_build $PHD/tools/docker/phd_build
```

Run it using `tools/docker/phd_build/run.sh`.

Update the dockerhob version using `tools/docker/phd_build/export.sh`.

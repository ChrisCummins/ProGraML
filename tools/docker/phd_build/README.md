# Docker image: phd

An Ubuntu environment configured for building this project. This is basically
the dockerfile version of the
[build instructions](https://github.com/ChrisCummins/phd#---building-the-code---------).

Build using:

```sh
$ docker build -t phd_build --squash $PHD/tools/docker/phd_build
```

# Docker image: phd

A self-contained environment configured to build and run the code in this 
project.

Build this image using:

```sh
$ docker build -t phd_build $PHD/tools/docker/phd_build
```

Run it using:

```sh
$ docker run -it -v/var/run/docker.sock:/var/run/docker.sock -v$HOME/.ssh:/root/.ssh phd_build
```

To update the image, run (from the host):

```sh
$ docker ps
# copy CONTAINER ID
$ docker commit CONTAINER phd_build_instance:latest
```

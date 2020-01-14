#!/usr/bin/env bash
#
# Build and publish updated docker image.
set -eux

version=$(cat version.txt)

docker build -t phd_base_java "$PHD"/tools/docker/phd_base_java
docker tag phd_base_java chriscummins/phd_base_java:latest
docker tag phd_base_java chriscummins/phd_base_java:"$version"
docker push chriscummins/phd_base_java:latest
docker push chriscummins/phd_base_java:"$version"
docker rmi phd_base_java

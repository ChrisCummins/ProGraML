#!/usr/bin/env bash
#
# Build and publish updated docker image.
set -eux

version=$(cat version.txt)

docker build -t phd_build "$PHD"/tools/docker/phd_build
docker tag phd_build chriscummins/phd_build:latest
docker tag phd_build chriscummins/phd_build:"$version"
docker push chriscummins/phd_build:latest
docker push chriscummins/phd_build:"$version"
docker rmi phd_build

#!/usr/bin/env bash
#
# Build and publish updated docker image.
set -eux

version=$(cat version.txt)

docker build -t phd_base "$PHD"/tools/docker/phd_base
docker tag phd_base chriscummins/phd_base:latest
docker tag phd_base chriscummins/phd_base:"$version"
docker push chriscummins/phd_base:latest
docker push chriscummins/phd_base:"$version"

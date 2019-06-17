#!/usr/bin/env bash
#
# Build and publish updated docker image.
set -eux

docker build -t phd_base $PHD/tools/docker/phd_base
docker tag phd_base chriscummins/phd_base:latest
docker push chriscummins/phd_base:latest

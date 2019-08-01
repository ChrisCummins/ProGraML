#!/usr/bin/env bash

cat config.pbtxt | grep -v '^#'
cat build_info.pbtxt | grep -v '^#'

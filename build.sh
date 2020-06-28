#!/bin/bash

set -eu

rm -rf build
mkdir -p build
cd build
cmake ..
make -j

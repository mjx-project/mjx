#!/bin/bash

set -eu

mkdir -p build
cd build
cmake ..
make

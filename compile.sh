#!/bin/bash
rm -rf build
mkdir -p build
pushd build
cmake ..
make
cp example ..
popd
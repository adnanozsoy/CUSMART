#!/bin/bash

cd ..
mkdir -p build
mkdir -p bin

cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . -- -j

cd ../scripts

#!/bin/bash
mkdir -p build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
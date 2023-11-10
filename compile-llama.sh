#!/bin/bash

cmake -S llama.cpp -B llama.cpp/build
cmake --build llama.cpp/build --config Release -- -j

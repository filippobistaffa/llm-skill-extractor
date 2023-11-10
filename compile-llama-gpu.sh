#!/bin/bash

cmake -S llama.cpp -B llama.cpp/build -DLLAMA_CUBLAS=ON
cmake --build llama.cpp/build --config Release -- -j

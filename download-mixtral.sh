#!/bin/bash

huggingface-cli download TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf --local-dir llama.cpp/models --local-dir-use-symlinks False

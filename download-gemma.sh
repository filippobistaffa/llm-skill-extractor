#!/bin/bash

huggingface-cli download rahuldshetty/gemma-7b-it-gguf-quantized gemma-7b-it-Q4_K_M.gguf --local-dir llama.cpp/models --local-dir-use-symlinks False

#!/bin/bash

echo "compiling cuda.cu..."
nvcc cuda.cu -o cuda

if [ $? -eq 0 ]; then
    echo "compilation successful. running ./cuda..."
    ./cuda
else
    echo "compilation failed."
fi
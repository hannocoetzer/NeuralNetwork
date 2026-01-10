#!/bin/bash

echo "compiling vector_add.cu..."
nvcc vector_add.cu -o vector_add

if [ $? -eq 0 ]; then
    echo "compilation successful. running ./vector_add..."
    ./vector_add
else
    echo "compilation failed."
fi
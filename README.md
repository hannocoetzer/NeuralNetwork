# Windows C++ / CUDA Setup Guide

1. Install g++
Download and install **MSYS2**:  
https://www.msys2.org/

After installation, install the MinGW toolchain and open msys2 terminal and run:
pacman -S mingw-w64-x86_64-gcc

2. Install NVIDIA CUDA Toolkit
Download and install CUDA:
https://developer.nvidia.com/cuda/toolkit

3. Install Visual C++ Build Tools
Download:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

During installation, make sure to select:

✅ Desktop development with C++

✅ MSVC v143 (or latest)

✅ Windows 10 SDK

You do NOT need the full Visual Studio IDE.

4. Add PATH Environment Variables
Add the following to your system PATH:


C:\msys64\mingw64\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin

5. Running Code
CUDA (GPU)
Open x64 Native Tools Command Prompt for VS 2022
nvcc cuda.cu -o cuda.exe
./cuda.exe

CPU (g++)

g++ neuralnetwork.cpp -o neuralnetwork.exe
./neuralnetwork.exe

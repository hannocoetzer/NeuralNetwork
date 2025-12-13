# C++ and CUDA Development Setup Guide

This guide covers two separate installation paths: one for CPU-based C++ development using g++, and another for GPU-accelerated development using NVIDIA CUDA Toolkit.

---

## Installation 1: CPU Development with g++

### Step 1: Install MSYS2

Download and install **MSYS2** from:  
https://www.msys2.org/

### Step 2: Install MinGW Toolchain

After MSYS2 installation completes, open the **MSYS2 terminal** and run:

```bash
pacman -S --needed base-devel mingw-w64-x86_64-gcc
```

### Step 3: Add to PATH

Add the following directory to your system PATH environment variable:

```
C:\msys64\mingw64\bin
```

### Step 4: Compile and Run

Open any terminal and use g++ to compile your code:

```bash
g++ neuralnetwork.cpp -o neuralnetwork.exe
./neuralnetwork.exe
```

---

## Installation 2: GPU Development with CUDA Toolkit

### Step 1: Install NVIDIA CUDA Toolkit

Download and install CUDA from:  
https://developer.nvidia.com/cuda-toolkit

### Step 2: Install Visual C++ Build Tools

Download Visual C++ Build Tools from:  
https://visualstudio.microsoft.com/visual-cpp-build-tools/

During installation, select the following components:

- ✅ Desktop development with C++
- ✅ MSVC v143 (or latest version)
- ✅ Windows 10 SDK

**Note:** You do NOT need the full Visual Studio IDE.

### Step 3: Add CUDA to PATH

Add the following directory to your system PATH environment variable (adjust version number as needed):

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
```

### Step 4: Compile and Run

Open **x64 Native Tools Command Prompt for VS 2022** and use nvcc to compile:

```bash
nvcc cuda.cu -o cuda.exe
./cuda.exe
```

---

## Quick Reference

| Development Type | Compiler | Command Prompt | Compile Command |
|-----------------|----------|----------------|-----------------|
| CPU (C++) | g++ | Any terminal | `g++ source.cpp -o output.exe` |
| GPU (CUDA) | nvcc | x64 Native Tools | `nvcc source.cu -o output.exe` |

---

## Additional Resources

**CUDA Learning:**
- [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) - NVIDIA Developer Blog

**Recommended Tutorial:**
- Jeff Heaton's CUDA tutorials and examples

#include <iostream>
#include <vector>

// CUDA kernel to add two arrays
__global__ void add(float *a, float *b, float *c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int size = 100; // Size of the arrays
    float *a, *b, *c; // Host arrays
    float *g_a, *g_b, *g_c; // Device arrays

    // Allocate memory on host
    a = (float *)malloc(size * sizeof(float));
    b = (float *)malloc(size * sizeof(float));
    c = (float *)malloc(size * sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < size; i++) {
        a[i] = i * 1.2;
        b[i] = i * 2.2;
    }

    // Allocate memory on device
    cudaMalloc((void **)&g_a, size * sizeof(float));
    cudaMalloc((void **)&g_b, size * sizeof(float));
    cudaMalloc((void **)&g_c, size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(g_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

    // Define kernel launch parameters
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    // Launch the kernel
    add<<<numBlocks, blockSize>>>(g_a, g_b, g_c, size);

    // Copy result from device to host
    cudaMemcpy(c, g_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print a few results to verify
    std::cout << "Vector Addition Result (first 10 elements):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    }

    // Free device memory
    cudaFree(g_a);
    cudaFree(g_b);
    cudaFree(g_c);

    // Free host memory
    free(a);
    free(b);
    free(c);

}
#include <iostream>
#include <vector>

// CUDA kernel to add two arrays
__global__ void add(int *a, int *b, int *sum, int *mul, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        sum[tid] = a[tid] + b[tid];
        mul[tid] = a[tid] * b[tid];
    }
}

int main() {
    int size = 100; // Size of the arrays
    int *a, *b, *sum, *mul; // Host arrays
    int *g_a, *g_b, *g_sum, *g_mul; // Device arrays

    // Allocate memory on host
    a = (int *)malloc(size * sizeof(int));
    b = (int *)malloc(size * sizeof(int));
    sum = (int *)malloc(size * sizeof(int));
    mul = (int *)malloc(size * sizeof(int));


    // Initialize host arrays
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Allocate memory on device
    cudaMalloc((void **)&g_a, size * sizeof(int));
    cudaMalloc((void **)&g_b, size * sizeof(int));
    cudaMalloc((void **)&g_sum, size * sizeof(int));
    cudaMalloc((void **)&g_mul, size * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(g_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Define kernel launch parameters
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    // Launch the kernel
    add<<<numBlocks, blockSize>>>(g_a, g_b, g_sum,g_mul, size);

    // Copy result from device to host
    cudaMemcpy(sum, g_sum, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(mul, g_mul, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print a few results to verify
    std::cout << "Vector Addition Result (first 10 elements):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << a[i] << " + " << b[i] << " = " << sum[i] << std::endl;
        std::cout << a[i] << " * " << b[i] << " = " << mul[i] << std::endl;
    }

    // Free device memory
    cudaFree(g_a);
    cudaFree(g_b);
    cudaFree(g_sum);
    cudaFree(g_mul);

    // Free host memory
    free(a);
    free(b);
    free(sum);
    free(mul);

}
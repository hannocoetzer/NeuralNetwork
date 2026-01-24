#include <iostream>
#include <vector>

// Struct containing 2 float variables
struct Node {
    float x;
    float y;
};

// CUDA kernel for element-wise vector multiplication
__global__ void vectorMultiply(Node* a, Node* b, Node* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        result[idx].x = a[idx].x * b[idx].x;
        result[idx].y = a[idx].y * b[idx].y;
    }
}

int main() {
    const int N = 1024;
    const int size = N * sizeof(Node);
    
    // Allocate host memory
    Node* h_a = (Node*)malloc(size);
    Node* h_b = (Node*)malloc(size);
    Node* h_result = (Node*)malloc(size);
    
    // Initialize input arrays
    for (int i = 0; i < N; i++) {
        h_a[i].x = i * 1.0f;
        h_a[i].y = i * 2.0f;
        h_b[i].x = 2.0f;
        h_b[i].y = 3.0f;
    }
    
    // Allocate device memory
    Node *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_result, size);
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, N);
    
    // Copy result back to host
    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
    
    // Print first 5 results
    printf("First 5 results:\n");
    for (int i = 0; i < 5; i++) {
        printf("result[%d] = (%.2f, %.2f)\n", i, h_result[i].x, h_result[i].y);
    }
    
    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    free(h_a);
    free(h_b);
    free(h_result);
    
    return 0;
}
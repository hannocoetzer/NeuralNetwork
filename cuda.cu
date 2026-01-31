#include <iostream>
#include <vector>
#include <cooperative_groups.h>
#include <stdio.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

enum nodeType{

};

enum layerType{
    NORMAL,
    HIDDEN,
    LINK
};

// Struct containing 2 float variables
struct node {
    float data;
    float ideal;
    float sum;
    float dfSum;
};

struct link{
    float weight;
    float weight_adjustment;
    float momentum_multiplier;
    float gradient;
    float gradient_total;

};

enum Variable
{
    data,
    sum,
    dfSum,
    weight
};

class layer{

    public:
        layerType layer_Type;
        int size;
        int bias_Count = 0;
        node* c_node_arr;
        node* g_node_arr;
        link* c_link_arr;
        link* g_link_arr;

        layer(int _size,int _bias_Count, layerType _layer_Type)
        {
            size = _size;            
            layer_Type = _layer_Type;

            if(layer_Type == NORMAL)
            {
                bias_Count = _bias_Count;
            }
            if(layer_Type == LINK)
            {
                bias_Count = 0;
            }
        }
        void alloc(){
            
        }
        void init(){
      
            for (int i = 0; i < size + bias_Count; i++) {

                if(layer_Type == NORMAL)
                {
                    c_node_arr[i].data = (i + 2) * 1.0f;
                    c_node_arr[i].sum = 0;
                    //c_node_arr[i].ideal = (i + 3) * 2.0f;
                }
                if(layer_Type == LINK)
                {
                    c_link_arr[i].weight = (i + 2) * 1.0f;
                    //c_link_arr[i].weight_adjustment = (i + 3) * 2.0f;
                }
            }

        }
        void print(Variable _var){
            printf("\n[");
            for (int i = 0; i < size + bias_Count; i++) {
                if(_var == data)
                    printf(" %.2f ,", c_node_arr[i].data);
                if(_var == sum)
                    printf(" %.2f ,", c_node_arr[i].sum);
                if(_var == dfSum)
                    printf(" %.2f ,", c_node_arr[i].dfSum);
                if(_var == weight)
                    printf(" %.2f ,", c_link_arr[i].weight);
            }
            printf("]");
        }
        void c_malloc(){
            if(layer_Type == LINK){
                c_link_arr = (link*)malloc(size * sizeof(link));
            }
            if(layer_Type == NORMAL){
                printf("c_malloc");
                c_node_arr = (node*)malloc(size * sizeof(node) + bias_Count * sizeof(node));
            }
        }
        void g_malloc(){
            if(layer_Type == LINK){
                cudaMalloc(&g_link_arr,sizeof(link)*size);
            }
            if(layer_Type == NORMAL){
                printf("g_malloc");
                cudaMalloc(&g_node_arr,sizeof(node)*size + sizeof(node)*bias_Count);
            }
        }
        void c_to_g(){
            if(layer_Type == LINK){
                cudaMemcpy(g_link_arr, c_link_arr, sizeof(link)*size, cudaMemcpyHostToDevice);
            }
            if(layer_Type == NORMAL){
                printf("cudaMemcpy");
                cudaMemcpy(g_node_arr, c_node_arr, sizeof(node)*size + sizeof(node)*bias_Count, cudaMemcpyHostToDevice);
            }
        }
        void g_to_c(){
            if(layer_Type == LINK){
                cudaMemcpy(c_link_arr, g_link_arr, sizeof(link)*size, cudaMemcpyDeviceToHost);
            }
            if(layer_Type == NORMAL){
                printf("g_to_c()");
                cudaMemcpy(c_node_arr, g_node_arr, sizeof(node)*size  + sizeof(node)*bias_Count, cudaMemcpyDeviceToHost);
            }
        }
        void del(){
            if(layer_Type == LINK){
                free(c_link_arr);
                cudaFree(g_link_arr);
            }
            if(layer_Type == NORMAL){
                printf("del");
                free(c_node_arr);
                cudaFree(g_node_arr);
            }
        }


};

// CUDA kernel for element-wise vector multiplication
__global__ void vectorMultiply(node* a, node* b, node* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        result[idx].data = a[idx].data * b[idx].data;
        result[idx].ideal = a[idx].ideal + b[idx].ideal;
    }
}

__global__ void vectorSum(node* a, node* b, node* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        result[idx].data = a[idx].data * b[idx].data;
        result[idx].ideal = a[idx].ideal + b[idx].ideal;
    }
}

__device__ float sigmoid(float sum)
{
    return  1.0f / (1.0f + expf(-1 * sum));

}

__device__ float sigmoidDerivative(float x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

__global__ void forwardProp(node* a, link* b, node* result, int sizeA, int numSubsets) {

    // a [1,2,3]
    // b [3,4,5,6,7,8]
    
    //lock grid or something

    cg::grid_group grid = cg::this_grid();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int subSetSize = sizeA;         //3 => see below why this is should be the same
    int totalOps = numSubsets * sizeA; //6 Ops => 2 subsets -> each element in a subSet (of B) needs to multiplied by each element in A
    
    //will not go past 6
    if (idx < totalOps) {

        int idxA = idx % sizeA;         // 6 Ops => idx shouldn't go greater than > 3        
        int subsetIdx = idx / sizeA;    // 6 Ops is split into 2 parts = 6 ops / 3 size = 2 threads to multiply

        // you could almost see this as another thread - maybe even if(subsetIdx < totalOps / sizeA) or if(subsetIdx < subSetSize)
        int idxB = subsetIdx *          // subsetIdx will not go past 2
                        subSetSize      // just to give it a size or something
                            + idxA;     // idxA is the incrementor and will not go past 3 - can be seen as (for idxA = 0; idx % sizeA ; idxA ++)

        float product = b[idxB].weight * a[idxA].data;
        result[idx].data = product;
        result[idx].sum = product;
        atomicAdd(&result[idx].sum, product);
    }

    // //grid sync !!! when not enabled return 0, else return values
    grid.sync();

    // //we just want to do a quick activation sigmoid calculation
    // if(idx < numSubsets)
    // {
    //     result[idx].data = result[idx].data * 1.2;
    //     //result[idx].data = sigmoid(result[idx].sum);
    //     //result[idx].dfSum = sigmoidDerivative(result[idx].sum);
    // }
}

int main() {
    // const int N = 1024;
    // const int size = N * sizeof(node);
    
    // // Allocate host memory
    // node* h_a = (node*)malloc(size);
    // node* h_b = (node*)malloc(size);
    // node* h_result = (node*)malloc(size);
    
    // // Initialize input arrays
    // for (int i = 0; i < N; i++) {
    //     h_a[i].data = i * 1.0f;
    //     h_a[i].ideal = i * 2.0f;
    //     h_b[i].data = 2.0f;
    //     h_b[i].ideal = 3.0f;
    // }
    
    // // Allocate device memory
    // node *d_a, *d_b, *d_result;
    // cudaMalloc(&d_a, size);
    // cudaMalloc(&d_b, size);
    // cudaMalloc(&d_result, size);
    
    // // Copy data to device
    // cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // // Launch kernel
    // int threadsPerBlock = 256;
    // int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // vectorMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, N);
    
    // // Copy result back to host
    // cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
    
    // // Print first 5 results
    // printf("First 5 results:\n");
    // for (int i = 0; i < 5; i++) {
    //     printf("result[%d] = (%.2f, %.2f)\n", i, h_result[i].data, h_result[i].ideal);
    // }
    
    // // Free memory
    // cudaFree(d_a);
    // cudaFree(d_b);
    // cudaFree(d_result);
    // free(h_a);
    // free(h_b);
    // free(h_result);

    layer* i1 = new layer(2,1,NORMAL);
    i1->c_malloc();
    i1->init();
    i1->g_malloc();

    layer* h1 = new layer(2,0,NORMAL);
    h1->c_malloc();
    h1->init();
    h1->g_malloc();

    printf("\n\ni1->size + i1->bias_Count : %i", i1->size + i1->bias_Count);
    printf("\nh1->size : %i", h1->size);
    printf("\n(i1->size + i1->bias_Count) * (h1->size) : %i\n\n", (i1->size + i1->bias_Count) * (h1->size));

    layer* w1 = new layer((i1->size + i1->bias_Count) * (h1->size),0,LINK);
    w1->c_malloc();
    w1->init();
    w1->g_malloc();

    i1->print(data);
    w1->print(weight);
    h1->print(data);

    int threadsPerBlock = 256;
    int opSize = w1->size;
    int blocksPerGrid = (opSize + threadsPerBlock - 1) / threadsPerBlock;

    printf("\n\nopSize : %i", opSize);
    printf("\ni1->size + i1->bias_Count : %i", i1->size + i1->bias_Count);
    printf("\nw1->size : %i", w1->size);
    printf("\nnumOfubsets : %i\n\n", w1->size / (i1->size + i1->bias_Count));
    forwardProp<<<blocksPerGrid, threadsPerBlock>>>(i1->g_node_arr, w1->g_link_arr, h1->g_node_arr, i1->size + i1->bias_Count, w1->size / (i1->size + i1->bias_Count));
    cudaDeviceSynchronize();
    
    h1->g_to_c();
    i1->print(data);
    h1->print(data);

    i1->del();
    h1->del();
    w1->del();
    

    // layer* o = new layer(1,0,NORMAL);
    // o->c_malloc();
    // o->g_malloc();

    // i1->c_to_g();
    // h1->c_to_g();


    /*int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // -- Output saved to new node* array
    // vectorMultiply<<<blocksPerGrid, threadsPerBlock>>>(i1->g_node_arr, h1->g_node_arr, o->g_node_arr, N);
    // o->g_to_c();
    // i1->print();
    // h1->print();
    // o->print();

    // -- Output overwrite current node* array
    vectorMultiply<<<blocksPerGrid, threadsPerBlock>>>(i1->g_node_arr, h1->g_node_arr, h1->g_node_arr, N);
    i2->g_to_c();
    i1->print();
    i2->print();

    //clean up
    i1->del();
    i2->del();
    o->del();*/
    
    return 0;
}
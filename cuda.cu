#include <iostream>
#include <vector>

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

class layer{

    public:
        layerType layer_Type;
        int size;
        int bias_Count;
        node* c_node_arr;
        node* g_node_arr;
        link* c_link_arr;
        link* g_link_arr;

        layer(int _size,int _bias_Count, layerType _layer_Type) : size(_size),bias_Count(_bias_Count),layer_Type(_layer_Type){

        }
        void alloc(){
            
        }
        void init(){
      
            for (int i = 0; i < size; i++) {
                c_node_arr[i].data = (i + 2) * 1.0f;
                c_node_arr[i].ideal = (i + 3) * 2.0f;
            }

        }
        void print(){
            for (int i = 0; i < size; i++) {
                printf("result[%d] = (%.2f, %.2f)\n", i, c_node_arr[i].data, c_node_arr[i].ideal);
            }
        }
        void c_malloc(){
            if(layer_Type == LINK){
                c_link_arr = (link*)malloc(size * sizeof(link));
            }
            if(layer_Type == NORMAL){
                printf("c_malloc");
                c_node_arr = (node*)malloc(size * sizeof(node));
            }
        }
        void g_malloc(){
            if(layer_Type == LINK){
                cudaMalloc(&g_link_arr,sizeof(link)*size);
            }
            if(layer_Type == NORMAL){
                printf("g_malloc");
                cudaMalloc(&g_node_arr,sizeof(node)*size);
            }
        }
        void c_to_g(){
            if(layer_Type == LINK){
                cudaMemcpy(g_link_arr, c_link_arr, sizeof(link)*size, cudaMemcpyHostToDevice);
            }
            if(layer_Type == NORMAL){
                printf("cudaMemcpy");
                cudaMemcpy(g_node_arr, c_node_arr, sizeof(node)*size, cudaMemcpyHostToDevice);
            }
        }
        void g_to_c(){
            if(layer_Type == LINK){
                cudaMemcpy(c_link_arr, g_link_arr, sizeof(link)*size, cudaMemcpyDeviceToHost);
            }
            if(layer_Type == NORMAL){
                printf("g_to_c()");
                cudaMemcpy(c_node_arr, g_node_arr, sizeof(node)*size, cudaMemcpyDeviceToHost);
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

    int N = 2;

    layer* i1 = new layer(N,0,NORMAL);
    i1->c_malloc();
    i1->init();
    i1->g_malloc();

    layer* i2 = new layer(N,0,NORMAL);
    i2->c_malloc();
    i2->init();
    i2->g_malloc();

    layer* o = new layer(N,0,NORMAL);
    o->c_malloc();
    o->g_malloc();

    i1->c_to_g();
    i2->c_to_g();


    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorMultiply<<<blocksPerGrid, threadsPerBlock>>>(i1->g_node_arr, i2->g_node_arr, o->g_node_arr, N);

    o->g_to_c();
    i1->print();
    i2->print();
    o->print();

    i1->del();
    i2->del();
    o->del();
    
    return 0;
}
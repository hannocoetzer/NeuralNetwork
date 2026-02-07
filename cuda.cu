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

struct node {
    float data;
    float ideal;
    float sum;
    float dfSum;
    float delta;
    float sumOfWeights;
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
        void c_to_g(bool copyWithBias){
            if(layer_Type == LINK){
                printf("\nL_c_to_g");
                cudaMemcpy(g_link_arr, c_link_arr, sizeof(link)*size, cudaMemcpyHostToDevice);
            }
            if(layer_Type == NORMAL){
                printf("\nN_c_to_g");
                // faulty for some reason
                //cudaMemcpy(g_node_arr, c_node_arr, sizeof(node)*size + copyWithBias ? sizeof(node)*bias_Count : 0, cudaMemcpyHostToDevice);
                cudaMemcpy(g_node_arr, c_node_arr, sizeof(node)*size + sizeof(node)*bias_Count, cudaMemcpyHostToDevice);
            }
        }
        void g_to_c(bool copyWithBias){
            if(layer_Type == LINK){
                printf("\nL_g_to_c");
                cudaMemcpy(c_link_arr, g_link_arr, sizeof(link)*size, cudaMemcpyDeviceToHost);
            }
            if(layer_Type == NORMAL){
                printf("\nN_g_to_c");
                // faulty for some reason
                //cudaMemcpy(c_node_arr, g_node_arr, sizeof(node)*size  + copyWithBias ? sizeof(node)*bias_Count : 0, cudaMemcpyDeviceToHost);
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

    // ex. a [1,2,3]
    // ex. b [3,4,5,6,7,8]
    // ex. result [(3 * 1) + (4 * 2) + (4 * 3), (3 * 1) + (4 * 2) + (4 * 3)]
    
    //lock grid or something
    //cg::grid_group grid = cg::this_grid();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int subSetSize = sizeA;         //3 => see below why this is should be the same
    int totalOps = numSubsets * sizeA; //6 Ops => 2 subsets -> each element in a subSet (of B) needs to multiplied by each element in A
    
    //will not go past 6
    if (idx < totalOps) {

        int idxA = idx % sizeA;         // 6 Ops => idx shouldn't go greater than > 3        
        int subsetIdx = idx / sizeA;    // 6 Ops is split into 2 parts = 6 ops / 3 size = 2 threads to multiply

        printf("\nidxA : %i",idxA);
        printf("\nsubsetIdx : %i",subsetIdx);

        // you could almost see this as another thread - maybe even if(subsetIdx < totalOps / sizeA) or if(subsetIdx < subSetSize)
        int idxB = subsetIdx *          // subsetIdx will not go past 2
                        subSetSize      // just to give it a size or something
                            + idxA;     // idxA is the incrementor and will not go past 3 - can be seen as (for idxA = 0; idx % sizeA ; idxA ++)

        printf("\nidxB : %i",idxB);
        printf("\nb[idxB].weight : %0.2f",b[idxB].weight);
        printf("\na[idxA].data : %0.2f",a[idxA].data);

        float product = b[idxB].weight * a[idxA].data;
        printf("\nproduct : %0.2f",product);
        atomicAdd(&result[subsetIdx].sum, product); //important [subsetIdx] NOT [idx] - because we want to get the subset total
    }

    // grid sync !!! when not enabled return 0, else return values
    //grid.sync();

    //we just want to do a quick activation sigmoid calculation
    if(idx < numSubsets)
    {
        printf("\nidx : %i",idx);
        printf("\nresult[idx].sum: %0.2f",result[idx].sum);
        result[idx].data = result[idx].sum;
        //result[idx].data = sigmoid(result[idx].sum);
        //result[idx].dfSum = sigmoidDerivative(result[idx].sum);
    }
}

__global__ void backwardProp(node* i, link* w, node* o, node* result, int sizeI, int sizeO, int numSubsets) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //calculate error and delta
    if(idx < sizeO){
        o[idx].delta = (-1) * (o[idx].data - o[idx].ideal) * o[idx].dfSum;
    }

    //calculate sumOfWeights
    if(idx < sizeI ){

        float valI = i[idx].sumOfWeights;

        int idxI = idx;
        int idxW = idx + sizeI;

        if(idxI < sizeI * sizeO && idxW < sizeI * sizeO){
            float sumOfWeights = i[idxI].sumOfWeights + w[idxW].weight;
            atomicAdd(&i[idx].sumOfWeights, sumOfWeights);
        }
    }

    // link->node->delta = layerNode->dfSum * sumOfWeight * link->node->delta;
    // link->props->gradient = (layerNode->data * link->node->delta);
    // link->props->gradientTotal = link->props->gradientTotal + link->props->gradient;
    //calculate gradient
    if(idx < sizeI * sizeO){
        int idxI = idx%sizeI;
        int idxO = idx%sizeO;
        int idxW = idxI + sizeI;

        o[idxO].delta = i[idxI].dfSum * i[idxI].sumOfWeights * o[idxO].delta;
        w[idx].gradient = i[idxI].data * o[idxO].delta;
        w[idx].gradient_total = w[idx].gradient_total + w[idx].gradient;
    }   

}

int main() {

    cudaDeviceReset();

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

    i1->c_to_g(true);
    h1->c_to_g(false);
    w1->c_to_g(false);

    i1->print(data);
    w1->print(weight);
    h1->print(data);

    int threadsPerBlock = 256;
    int opSize = w1->size;
    int blocksPerGrid = (opSize + threadsPerBlock - 1) / threadsPerBlock;

    printf("\n\nopSize : %i", opSize);
    printf("\n\nblocksPerGrid : %i", blocksPerGrid);
    printf("\ni1->size + i1->bias_Count : %i", i1->size + i1->bias_Count);
    printf("\nw1->size : %i", w1->size);
    printf("\nnumOfubsets : %i\n\n", w1->size / (i1->size + i1->bias_Count));
    int numOfSubsets = w1->size / (i1->size + i1->bias_Count);
    int size = i1->size + i1->bias_Count;
    forwardProp<<<blocksPerGrid, threadsPerBlock>>>(i1->g_node_arr, w1->g_link_arr, h1->g_node_arr, size, numOfSubsets);
    cudaDeviceSynchronize();
    
    h1->g_to_c(false);

    i1->print(data);
    w1->print(weight);
    h1->print(data);

    i1->del();
    h1->del();
    w1->del();
    
    return 0;
}
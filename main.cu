#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>
#include "morton.cu"
#ifndef EPSILON
#define EPSILON (0.0001)
#endif
#define ASCENDING (1)
#define DESCENDING (0)
#define K_NEAREST (2)
using namespace std;


void handleError(cudaError_t err, int line) {
    if (err != cudaSuccess) {
        printf("Cuda Error %d", line);
        exit(EXIT_FAILURE);
    }
}

template<typename T>
string convert_to_binary_string(const T value, bool skip_leading_zeroes = false)
{
    string str;
    bool found_first_one = false;
    const int bits = sizeof(T) * 8;  // Number of bits in the type

    for (int current_bit = bits - 1; current_bit >= 0; current_bit--)
    {
        if ((value & (1ULL << current_bit)) != 0)
        {
            if (!found_first_one)
                found_first_one = true;
            str += '1';
        }
        else
        {
            if (!skip_leading_zeroes || found_first_one)
                str += '0';
        }
    }

    return str;
}


__global__
void compactPoints(uint4 *values, uint64_t *mortons, uint32_t *prefixQueryIndex, uint4 *data, uint32_t *queryIndices, int numElements) {
    uint32_t i = (uint32_t) blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numElements) {
        int isQuery = mortons[i] & 1; //check LSD
        if(isQuery) {
            uint32_t nthQ = prefixQueryIndex[i];
            uint32_t qi = i-nthQ+1;
            queryIndices[nthQ-1] = qi;
        } else {
            uint32_t numQueriesToLeft = prefixQueryIndex[i];
            uint32_t di = i-numQueriesToLeft;
            data[di] = values[i];
        }
    }
}

__device__
uint32_t isqrt(uint32_t n) {
    register uint32_t root, remainder, place;
  
    root = 0;
    remainder = n;
    place = 0x40000000;
    while (place > remainder) {
        place = place >> 2;
    }
    
    while (place) {
        if (remainder >= root + place) {  
            remainder = remainder - root - place;
            root = root + (place << 1);
        }
        
        root = root >> 1;
        place = place >> 2;
    }
    
    return root;  
} 

__device__ 
uint32_t idistance(uint4 a, uint4 b) {
    uint32_t x = b.x-a.x;
    uint32_t y = b.y-a.y;
    uint32_t z = b.z-a.z;
    uint32_t n = x*x + y*y + z*z;
    return isqrt(n);
}

__device__ inline
void Comparator(uint64_t &a, uint64_t &b, uint32_t dir) {
    uint64_t t;
    if((a > b) == dir) {
        //Swap if a > b;
        t = a;
        a = b;
        b = t;
    }
}


__device__ 
void bitonicSort(uint64_t *values, uint32_t i)
{
    const uint32_t dir = ASCENDING;
    const uint32_t length = (K_NEAREST << 1);
    for (uint32_t size = 2; size < length; size <<= 1)
    {
        //Bitonic merge
        uint32_t ddd = dir ^ ((i & (size / 2)) != 0);

        for (uint32_t stride = K_NEAREST; stride > 0; stride >>= 1)
        {
            __syncthreads();
            uint32_t pos = 2 * i - (i & (stride - 1));
            Comparator(
                values[pos +      0],
                values[pos + stride],
                ddd
            );
        }
    }

    //ddd == dir for the last bitonic merge step
    {
        for (uint32_t stride = K_NEAREST; stride > 0; stride >>= 1)
        {
            __syncthreads();
            uint32_t pos = 2 * i - (i & (stride - 1));
            Comparator(
                values[pos +      0],
                values[pos + stride],
                dir
            );
        }
    }
}


__global__
void approxNearest(uint32_t *queryIndices, uint4 *values, uint4 *data, uint64_t *currentNearest, int numQueries, int numData) {
    __shared__ uint64_t candidates[2*K_NEAREST];
    uint32_t q = (uint32_t) blockIdx.x;
    if(q < numQueries) {
        uint32_t iq  = queryIndices[q]; // data point right to query point
        uint4 querypoint = values[q+iq];
        uint32_t i = (uint32_t) threadIdx.x;
        if(i < K_NEAREST) {
           iq = min(numData-K_NEAREST, max(K_NEAREST-1, iq));
           uint32_t leftidx = iq-i;
           uint32_t rightidx = iq+i+1;
           uint4 left = data[leftidx];
           uint4 right = data[rightidx];
           uint32_t leftdist = idistance(querypoint, left);          
           uint32_t rightdist = idistance(querypoint, right);          
           candidates[2*i] = ((uint64_t) leftdist << 32) | leftidx;
           candidates[2*i+1] = ((uint64_t) rightdist << 32) | rightidx;
        }
        __syncthreads();
        bitonicSort(candidates, i);
        __syncthreads();
        //Write to global memory
        if(i < K_NEAREST) {
            currentNearest[q*K_NEAREST+i] = candidates[i];
        }
    }
}

void findCandidates(uint32_t *queryIndices, uint4 *values, uint4 *data, uint64_t *nearest, int numQueries, int numData) {
    int threadsPerBlock = K_NEAREST;
    int blocksPerGrid = numQueries;
    approxNearest<<<blocksPerGrid, threadsPerBlock>>>(queryIndices, values, data, nearest, numQueries, numData);

    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    handleError(err, __LINE__);
}

int getMortons(float3 *devValues, uint4 *devIntValues, uint64_t *devMortons,
 const float xlen, const float ylen, const float zlen, const float maxlen,
 const int numElements, const int numData, const int numQueries) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    scalePoints<<<blocksPerGrid, threadsPerBlock>>>(devValues, devIntValues, numElements, xlen, ylen, zlen, maxlen);
    computeMortons<<<blocksPerGrid, threadsPerBlock>>>(devIntValues, devMortons, numData, numQueries);

    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    handleError(err, __LINE__);
    return EXIT_SUCCESS;
}

int pointCompaction(uint4 *devIntValues, uint64_t *devMortons, uint32_t *devPrefixQueryIndex, uint4 *devData, uint32_t *devQueryIndices, int numElements) {
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    compactPoints<<<blocksPerGrid, threadsPerBlock>>>(devIntValues, devMortons, devPrefixQueryIndex, devData, devQueryIndices, numElements);
    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    handleError(err, __LINE__);
    return EXIT_SUCCESS;
}

__global__
void prefixList(uint4 *values, uint32_t *prefix, int numData,  int numElements) {
    uint32_t i = (uint32_t) blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numElements) {
        if(values[i].w < numData) {
            prefix[i] = 0;
        } else {
            prefix[i] = 1;
        }
    }
}

int createPrefixList(uint4 *devIntValues, uint32_t *devPrefixQueryIndex, int numData, int numElements) {
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    prefixList<<<blocksPerGrid, threadsPerBlock>>>(devIntValues, devPrefixQueryIndex, numData, numElements);

    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    handleError(err, __LINE__);

    // in-place prefix sum on list
    thrust::inclusive_scan(thrust::device, devPrefixQueryIndex, devPrefixQueryIndex + numElements, devPrefixQueryIndex);
    err = cudaGetLastError();
    handleError(err, __LINE__);

    return EXIT_SUCCESS;
}

void initValues(float3 *values, float *xlen, float *ylen, float *zlen, float *maxlen, int numElements) {
    float randMax = (float) RAND_MAX;
    float minx = FLT_MAX;
    float maxx = FLT_MIN;
    float miny = FLT_MAX;
    float maxy = FLT_MIN;
    float minz = FLT_MAX;
    float maxz = FLT_MIN;
    for (int i = 0; i < numElements; ++i) {
        values[i].x = rand()/randMax;
        if(values[i].x < minx) {
            minx = values[i].x;
        }
        if(values[i].x > maxx) {
            maxx = values[i].x;
        }
        values[i].y = rand()/randMax;
        if(values[i].y < miny) {
            miny = values[i].y;
        }
        if(values[i].y > maxy) {
            maxy = values[i].y;
        }
        values[i].z = rand()/randMax;
        if(values[i].z < minz) {
            minz = values[i].z;
        }
        if(values[i].z > maxz) {
            maxz = values[i].z;
        }
    }
    float xl, yl, zl;
    xl = maxx-minx;
    yl = maxy-miny;
    zl = maxz-minz;
    *xlen = xl;
    *ylen = yl;
    *zlen = zl;
    *maxlen = fmax(xl,fmax(yl, zl));
}

int main(void)
{
    int numData = 5;
    int numQueries = 5;
    int numElements = numData + numQueries;

    assert(2*K_NEAREST <= numData);

    size_t valueSize = numElements * sizeof(float3);
    float3 *values = (float3 *) malloc(valueSize);

    size_t intSize = numElements * sizeof(uint4);
    uint4 *intValues = (uint4 *) malloc(intSize);

    size_t dataSize = numData * sizeof(uint4);
    uint4 *data = (uint4 *) malloc(dataSize);

    size_t qiSize = numQueries * sizeof(uint32_t);
    uint32_t *queryIndices = (uint32_t *) malloc(qiSize);

    size_t nearestSize = numQueries * K_NEAREST * sizeof(uint64_t);
    uint64_t *nearest = (uint64_t *) malloc(nearestSize);

    size_t mortonSize = numElements * sizeof(uint64_t);
    uint64_t *mortons = (uint64_t *) malloc(mortonSize);

    size_t prefixSize = numElements * sizeof(uint32_t);

    cudaError_t err = cudaSuccess;

    float3 *devValues = NULL;
    err = cudaMalloc((void **) &devValues, valueSize);
    handleError(err, __LINE__);

    uint4 *devIntValues = NULL;
    err = cudaMalloc((void **) &devIntValues, intSize);
    handleError(err, __LINE__);

    uint64_t *devMortons = NULL;
    err = cudaMalloc((void **) &devMortons, mortonSize);
    handleError(err, __LINE__);

    uint32_t *devPrefixQueryIndex = NULL;
    err = cudaMalloc((void **) &devPrefixQueryIndex, prefixSize);
    handleError(err, __LINE__);

    uint32_t *devQueryIndices = NULL;
    err = cudaMalloc((void **) &devQueryIndices, qiSize);
    handleError(err, __LINE__);

    uint4 *devData = NULL;
    err = cudaMalloc((void **) &devData, dataSize);
    handleError(err, __LINE__);

    uint64_t *devNearest = NULL;
    err = cudaMalloc((void **) &devNearest, nearestSize);
    handleError(err, __LINE__);

    float xlen, ylen, zlen, maxlen;
    initValues(values, &xlen, &ylen, &zlen, &maxlen, numElements);

    err = cudaMemcpy(devValues, values, valueSize, cudaMemcpyHostToDevice);
    handleError(err, __LINE__);
    
    getMortons(devValues, devIntValues, devMortons,
                    xlen, ylen, zlen, maxlen,
                    numElements, numData, numQueries);

    //sort values in morton code order
    thrust::sort_by_key(thrust::device, devMortons, devMortons + numElements, devIntValues);

    err = cudaMemcpy(intValues, devIntValues, intSize, cudaMemcpyDeviceToHost);
    handleError(err, __LINE__);

    err = cudaMemcpy(mortons, devMortons, mortonSize, cudaMemcpyDeviceToHost);
    handleError(err, __LINE__);
    
    createPrefixList(devIntValues, devPrefixQueryIndex, numData, numElements);

    for(int i = 0; i < numElements; ++i) {
        uint4 point = intValues[i];
        printf("%d: (%u,%u,%u) = %lu\n",
                point.w, 
                point.x,
                point.y,
                point.z,
                mortons[i]);      
    }
    printf("\n");

    pointCompaction(devIntValues, devMortons, devPrefixQueryIndex, devData, devQueryIndices, numElements);
    err = cudaMemcpy(queryIndices, devQueryIndices, qiSize, cudaMemcpyDeviceToHost);
    handleError(err, __LINE__);

    err = cudaMemcpy(data, devData, dataSize, cudaMemcpyDeviceToHost);
    handleError(err, __LINE__);

    for(int i = 0; i < numQueries; ++i) {
        printf("%u\n", queryIndices[i]);
    }

    for(int i = 0; i < numData; ++i) {
        uint4 point = data[i];
        printf("%d: (%u,%u,%u)\n", i, point.x, point.y, point.z);      
    }
    
    findCandidates(devQueryIndices, devIntValues, devData, devNearest, numQueries, numData);
    err = cudaMemcpy(nearest, devNearest, nearestSize, cudaMemcpyDeviceToHost);
    handleError(err, __LINE__);

    for(int i = 0; i < numQueries; ++i) {
       printf("[");
       for(int j = 0; j < K_NEAREST; ++j) {
               printf(" %u: ", (uint32_t) (nearest[i*K_NEAREST+j] >> 32));
               uint32_t point = (uint32_t) nearest[i*K_NEAREST+j]; //lower 32 bits
               printf(" %u ", point);
       }
       printf("], ");
       printf("[");
       for(int j = 0; j < K_NEAREST; ++j) {
               uint32_t point = (uint32_t) nearest[i*K_NEAREST+j]; //lower 32 bits
               printf(" (%u, %u, %u, %u) ", data[point].x,data[point].y,data[point].z, data[point].w);
       }
       printf("] ");
       uint4 querypoint = intValues[queryIndices[i]+i];
       printf("(%u, %u, %u, %u)\n", querypoint.x, querypoint.y, querypoint.z, querypoint.w);
    }

    err = cudaFree(devValues);
    handleError(err, __LINE__);

    err = cudaFree(devMortons);
    handleError(err, __LINE__);

    // Free host memory
    free(values);
    free(mortons);

    err = cudaDeviceReset();
    handleError(err, __LINE__);

    return EXIT_SUCCESS;
}

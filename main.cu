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
using namespace std;

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
void reorder(uint32_t *indices, uint32_t *input, uint32_t *output, int numElements) {
    uint32_t i = (uint32_t) blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numElements) {
        uint32_t j = indices[i];
        // swap
        output[i] = input[j];
    }
}

__global__
void compactPoints(uint32_t *values, uint64_t *mortons, uint32_t *indices, uint32_t *checkList, uint32_t *data, uint32_t *queryIndices, int numElements) {
    uint32_t i = (uint32_t) blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numElements) {
        int isQuery = mortons[i] & 1; //check LSD
        if(isQuery) {
            uint32_t nthQ = checkList[i];
            uint32_t qi = i-nthQ+1;
            queryIndices[nthQ-1] = qi;
        } else {
            uint32_t numQueriesToLeft = checkList[i];
            uint32_t di = i-numQueriesToLeft;
            uint32_t j = indices[i];
            data[di*3+0] = values[j*3+0];
            data[di*3+1] = values[j*3+1];
            data[di*3+2] = values[j*3+2];
        }
    }
}


__global__
void approxNearest(uint32_t *queryIndices, uint32_t *nearest, uint32_t k, int numQueries, int numData) {
    uint32_t i = (uint32_t) blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numQueries) {
        uint32_t iq  = queryIndices[i]; // data point right to query point
        iq = min(numData-k, max(k, iq));
        uint32_t minidx = iq-k;
        uint32_t maxidx = iq+k;
        uint32_t start = 2*k*i;
        uint32_t offset = 0;
        for(int a = minidx; a < maxidx; ++a) {
            nearest[start+offset] = a;
            offset += 1;
        }
    }
}

int main(void)
{
    cudaError_t err = cudaSuccess;
    int numData = 20;
    int numQueries = 5;
    uint32_t k = 4;
    assert(2*k <= numData);
    int numElements = numData + numQueries;
    size_t size = numElements * sizeof(float3);
    size_t intSize = 3 * numElements * sizeof(uint32_t);
    size_t indexSize = numElements * sizeof(uint32_t);
    size_t nearestSize = numQueries * 2 * k * sizeof(uint32_t);
    size_t resultSize = numElements * sizeof(uint64_t);
    size_t checkSize = numElements * sizeof(uint32_t);
    size_t dataSize = 3 * numData * sizeof(uint32_t);
    size_t qiSize = numQueries * sizeof(uint32_t);

    float3 *values = (float3 *) malloc(size);
    uint32_t *intValues = (uint32_t *) malloc(intSize);
    uint32_t *data = (uint32_t *) malloc(dataSize);
    uint32_t *queryIndices = (uint32_t *) malloc(qiSize);
    uint32_t *indices = (uint32_t *) malloc(indexSize);
    uint32_t *nearest = (uint32_t *) malloc(nearestSize);
    uint64_t *mortons = (uint64_t *) malloc(resultSize);
    uint32_t *checkList = (uint32_t *) malloc(checkSize);

    if (values == NULL || mortons == NULL) {
        exit(EXIT_FAILURE);
    }

    float randMax = (float) RAND_MAX;
    float minx = FLT_MAX;
    float maxx = FLT_MIN;
    float miny = FLT_MAX;
    float maxy = FLT_MIN;
    float minz = FLT_MAX;
    float maxz = FLT_MIN;
    for (int i = 0; i < numElements; ++i) {
        indices[i] = i;
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

        if(i < numData) {    
            checkList[i] = 0;
        } else {
            checkList[i] = 1;
        }
    }

    float3 *devValues = NULL;
    err = cudaMalloc((void **) &devValues, size);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }

    uint32_t *devIntValues = NULL;
    err = cudaMalloc((void **) &devIntValues, intSize);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }

    uint32_t *devIndices = NULL;
    err = cudaMalloc((void **) &devIndices, indexSize);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }

    uint64_t *devMortons = NULL;
    err = cudaMalloc((void **) &devMortons, resultSize);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }

    uint32_t *devCheckList = NULL;
    err = cudaMalloc((void **) &devCheckList, checkSize);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }

    uint32_t *devSortedCheckList = NULL;
    err = cudaMalloc((void **) &devSortedCheckList, checkSize);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }

    uint32_t *devQueryIndices = NULL;
    err = cudaMalloc((void **) &devQueryIndices, qiSize);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }

    uint32_t *devData = NULL;
    err = cudaMalloc((void **) &devData, dataSize);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }

    uint32_t *devNearest = NULL;
    err = cudaMalloc((void **) &devNearest, nearestSize);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(devValues, values, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(devIndices, indices, indexSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(devCheckList, checkList, checkSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }

    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    float xlen = maxx-minx;
    float ylen = maxy-miny;
    float zlen = maxz-minz;
    float maxlen = fmax(xlen,fmax(ylen, zlen));
    scalePoints<<<blocksPerGrid, threadsPerBlock>>>(devValues, devIntValues, numElements, xlen, ylen, zlen, maxlen);
    computeMortons<<<blocksPerGrid, threadsPerBlock>>>(devIntValues, devMortons, numData, numQueries);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(intValues, devIntValues, intSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(mortons, devMortons, resultSize, cudaMemcpyDeviceToHost);

    thrust::sort_by_key(thrust::device, devMortons, devMortons + numElements, devIndices);

    err = cudaMemcpy(mortons, devMortons, resultSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(indices, devIndices, indexSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }
    // reorder checkList by indices
    reorder<<<blocksPerGrid, threadsPerBlock>>>(devIndices, devCheckList, devSortedCheckList, numElements);
    err = cudaMemcpy(checkList, devSortedCheckList, checkSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }
    thrust::inclusive_scan(thrust::device, devSortedCheckList, devSortedCheckList + numElements, devSortedCheckList);

    err = cudaMemcpy(checkList, devSortedCheckList, checkSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }
    compactPoints<<<blocksPerGrid, threadsPerBlock>>>(devIntValues, devMortons, devIndices, devSortedCheckList, devData, devQueryIndices, numElements);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Cuda Error %d", err);
        exit(EXIT_FAILURE);
    }




    for(int i = 0; i < numElements; ++i) {
        uint32_t j = indices[i];
        printf("%d: (%u,%u,%u) = %lu\n",
                j, 
                intValues[j*3+0],
                intValues[j*3+1],
                intValues[j*3+2],
                mortons[i]);      
    }
    printf("\n");

    err = cudaMemcpy(queryIndices, devQueryIndices, qiSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(data, devData, dataSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < numQueries; ++i) {
        printf("%u\n", queryIndices[i]);
    }

    for(int i = 0; i < numData; ++i) {
        printf("%d: (%u,%u,%u)\n", i, data[i*3+0], data[i*3+1], data[i*3+2]);      
    }
    
    approxNearest<<<blocksPerGrid, threadsPerBlock>>>(devQueryIndices, devNearest, k, numQueries, numData);
    err = cudaMemcpy(nearest, devNearest, nearestSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < numQueries; ++i) {
        printf("[");
        for(int j = 0; j < 2*k; ++j) {
            printf(" %u ", nearest[i*2*k+j]);
        }
        printf("], ");
        printf("[");
        for(int j = 0; j < 2*k; ++j) {
            uint32_t point = nearest[i*2*k+j];
            printf(" (%u, %u, %u) ", data[point*3+0],data[point*3+1],data[point*3+2]);
        }
        printf("]\n");
    }


    err = cudaFree(devValues);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }

    err = cudaFree(devMortons);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(values);
    free(indices);
    free(mortons);

    err = cudaDeviceReset();
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}

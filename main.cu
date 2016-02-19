#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <thrust/sort.h>
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

int main(void)
{
    cudaError_t err = cudaSuccess;
    int numData = 5;
    int numQueries = 5;
    int numElements = numData + numQueries;
    size_t size = numElements * sizeof(float3);
    size_t indexSize = numElements * sizeof(uint32_t);
    size_t resultSize = numElements * sizeof(uint64_t);

    float3 *values = (float3 *) malloc(size);
    uint32_t *indices = (uint32_t *) malloc(indexSize);
    uint64_t *mortons = (uint64_t *) malloc(resultSize);
    if (values == NULL || mortons == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
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
    }

    float3 *devValues = NULL;
    err = cudaMalloc((void **) &devValues, size);
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

    err = cudaMemcpy(devValues, values, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(devIndices, indices, indexSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }

    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    float xlen = maxx-minx;
    float ylen = maxy-miny;
    float zlen = maxz-minz;
    float maxlen = fmax(xlen,fmax(ylen, zlen));
    computeMortons<<<blocksPerGrid, threadsPerBlock>>>(devValues, devMortons, numData, numQueries, xlen, ylen, zlen, maxlen);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(mortons, devMortons, resultSize, cudaMemcpyDeviceToHost);

    for(int i = 0; i < numElements; ++i) {
        printf("%d: (%f,%f,%f) = %lu\n", indices[i], values[i].x, values[i].y, values[i].z, mortons[i]);      
    }

    printf("\n\n");
    thrust::sort_by_key(thrust::device, devMortons, devMortons + numElements, devIndices);


    err = cudaMemcpy(indices, devIndices, indexSize, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < numElements; ++i) {
        uint32_t j = indices[i];
        printf("%d: (%f,%f,%f) = %lu\n", j, values[j].x, values[j].y, values[j].z, mortons[j]);      
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

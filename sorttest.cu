#include <stdio.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>
#include "radix.cu"
#ifndef EPSILON
#define EPSILON (0.0001)
#endif
#define BITS_PER_COORD (21)

__global__ void
vectorAdd(uint64_t *values, unsigned int *keys, unsigned int *result, int numElements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numElements)
    {
            radix_sort(keys, values);
        result[i] = keys[i];
        printf("Test %d %ld %d\n", keys[i], values[i], result[i]);
    }
}

int main(void)
{
    cudaError_t err = cudaSuccess;
    int numElements = 5;
    size_t size = numElements * sizeof(uint64_t);
    size_t keySize = numElements * sizeof(unsigned int);
    size_t resultSize = numElements * sizeof(unsigned int);
    printf("[Vector addition of %d elements]\n", numElements);

    unsigned int *keys = (unsigned int *)malloc(keySize);
    uint64_t *values = (uint64_t *)malloc(size);
    unsigned int *result = (unsigned int *)malloc(resultSize);

    for (int i = 0; i < numElements; ++i)
    {
        keys[i] = i;
    }
    values[0] = 6;
    values[1] = 2;
    values[2] = 8;
    values[3] = 1;
    values[4] = 9;

    for(int i = 0; i < numElements; ++i) {
        printf("%d: %ld\n", keys[i], values[i]);
    }

    uint64_t *deviceValues = NULL;
    err = cudaMalloc((void **)&deviceValues, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    unsigned int *deviceKeys = NULL;
    err = cudaMalloc((void **)&deviceKeys, keySize);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    unsigned int *deviceResult = NULL;
    err = cudaMalloc((void **)&deviceResult, resultSize);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(deviceValues, values, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(deviceKeys, keys, keySize, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(deviceValues, deviceKeys, deviceResult, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(result, deviceResult, resultSize, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < numElements; ++i) {
        printf("%d: %ld - %d\n", keys[i], values[i], result[i]);
    }
    err = cudaFree(deviceValues);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(deviceKeys);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(deviceResult);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(values);
    free(keys);
    free(result);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

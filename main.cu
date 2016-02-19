#include <stdio.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>
#ifndef EPSILON
#define EPSILON (0.0001)
#endif
#define BITS_PER_COORD (21)
using namespace std;

inline void morton3D_Decode_for(const uint64_t m, unsigned int& x, unsigned int& y, unsigned int& z){
    x = 0; y = 0; z = 0;
    unsigned int checkbits = (sizeof(uint64_t) <= 4) ? 10 : 21;

    for (uint64_t i = 0; i <= checkbits; ++i) {
        x |= (m & (1ull << 3 * i)) >> ((2 * i));
        y |= (m & (1ull << ((3 * i) + 1))) >> ((2 * i) + 1);
        z |= (m & (1ull << ((3 * i) + 2))) >> ((2 * i) + 2);
    }
} 

// method to seperate bits from a given integer 3 positions apart
__device__
inline uint64_t splitBy3(unsigned int a){
    uint64_t x = a & 0x1fffff; // we only look at the first 21 bits
    x = (x | x << 32) & 0x1f00000000ffff;  // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
    x = (x | x << 16) & 0x1f0000ff0000ff;  // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
    x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
    x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

__device__
inline uint64_t mortonEncode(unsigned int x, unsigned int y, unsigned int z){
    uint64_t answer = 0;
    answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
    return answer;
}

// Assumes: x \in [0,1]
__device__
inline unsigned int toInt(float x) {
    return (unsigned int) (x * (1<<BITS_PER_COORD));
}

__device__
inline float scaleValue(float x, float len, float maxlen) {
    return fmin(fmax(0.0, 0.75*(x + 0.5*len)/maxlen), 0.75);
}

__global__ void
vectorAdd(const float3 *A, const float3 *B, uint64_t *C, int numElements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = mortonEncode(
                        toInt(scaleValue(A[i].x, 10.0, 10.0)),
                        toInt(scaleValue(A[i].y, 10.0, 10.0)),
                        toInt(scaleValue(A[i].z, 10.0, 10.0))
        );
        
        //C[i] = A[i].x * B[i].x + A[i].y * B[i].y + A[i].z * B[i].z;
    }
}

int main(void)
{
    cudaError_t err = cudaSuccess;
    int numElements = 50;
    size_t size = numElements * sizeof(float3);
    size_t resultSize = numElements * sizeof(uint64_t);
    printf("[Vector addition of %d elements]\n", numElements);

    float3 *h_A = (float3 *)malloc(size);
    float3 *h_B = (float3 *)malloc(size);
    uint64_t *h_C = (uint64_t *)malloc(resultSize);

    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numElements; ++i)
    {
        h_A[i].x = rand()/(float)RAND_MAX;
        h_A[i].y = rand()/(float)RAND_MAX;
        h_A[i].z = rand()/(float)RAND_MAX;

        h_B[i].x = rand()/(float)RAND_MAX;
        h_B[i].y = rand()/(float)RAND_MAX;
        h_B[i].z = rand()/(float)RAND_MAX;
    }

    float3 *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float3 *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    uint64_t *d_C = NULL;
    err = cudaMalloc((void **)&d_C, resultSize);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, resultSize, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < numElements; ++i) {
        printf("(%f,%f,%f) = %ld\n", h_A[i].x, h_A[i].y, h_A[i].z, h_C[i]);      
    }
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

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

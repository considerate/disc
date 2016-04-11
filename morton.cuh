#ifndef __MORTON_CU
#define __MORTON_CU
#include <float.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <thrust/device_vector.h>
#include <stdint.h>
#define BITS_PER_COORD (21)
#include "float3math.cuh"

typedef thrust::pair< thrust::device_ptr<float3>, thrust::device_ptr<float3> > float3pair;
typedef thrust::pair< thrust::device_ptr<float3idx>, thrust::device_ptr<float3idx> > float3pairidx;

__device__ __host__ inline uint64_t mortonEncode(unsigned int x, unsigned int y, unsigned int z);
__global__ void computeMortons(const uint4 *values, uint64_t *mortons, int numData, int numQueries);
__global__ void computeMortons(float3idx *values, uint32_t shift, uint64_t *mortons, int numData, int numQueries);
__global__ void scalePoints(const float3 *values, uint32_t *indices, uint4 *intValues, int numElements, uint32_t intShift, float3pair xpair, float3pair ypair, float3pair zpair);
__global__ void scalePoints(float3idx *values, float3idx *output, int numElements, float3 mins, float maxlen, bool revert);
__global__ void scalePointsOld(float3 *values, uint4 *intValues, int numElements, uint32_t intShift, float3 mins, float maxlen, bool revert);
#endif

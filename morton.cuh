#ifndef __MORTON_CU
#define __MORTON_CU
#include <float.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <thrust/device_vector.h>
#include <stdint.h>
#define BITS_PER_COORD (21)

typedef thrust::pair< thrust::device_ptr<float3>, thrust::device_ptr<float3> > float3pair;

__global__ void computeMortons(const uint4 *values, uint64_t *mortons, int numData, int numQueries);
__global__ void scalePoints(const float3 *values, uint32_t *indices, uint4 *intValues, int numElements, uint32_t intShift, float3pair xpair, float3pair ypair, float3pair zpair);
__global__ void scalePointsOld(const float3 *values, uint4 *intValues, int numElements, uint32_t intShift, float minx, float miny, float minz, float maxlen);
void scaleValues(float3 *values, int numElements, float normalScale, float tangentScale);
#endif

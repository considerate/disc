#ifndef __MORTON_CU
#define __MORTON_CU
#include <float.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <stdint.h>
#define BITS_PER_COORD (21)
__global__ void computeMortons(const uint4 *values, uint64_t *mortons, int numData, int numQueries);
__global__ void scalePoints(const float3 *values, uint4 *intValues, int numElements, uint32_t intShift, float minx, float miny, float minz, float maxlen);
#endif

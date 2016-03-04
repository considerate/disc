#ifndef __MORTON_CU
#define __MORTON_CU
#include <float.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#define BITS_PER_COORD (21)
inline void morton3D_Decode_for(const uint64_t m, unsigned int& x, unsigned int& y, unsigned int& z){
    x = 0; y = 0; z = 0;
    unsigned int checkbits = (sizeof(uint64_t) <= 4) ? 10 : 21;

    for (uint64_t i = 0; i <= checkbits; ++i) {
        x |= (m & (1ull << 3 * i)) >> ((2 * i));
        y |= (m & (1ull << ((3 * i) + 1))) >> ((2 * i) + 1);
        z |= (m & (1ull << ((3 * i) + 2))) >> ((2 * i) + 2);
    }
} 

__device__ __host__
inline uint64_t splitBy3(unsigned int a){
    uint64_t x = a & 0x1fffff; // we only look at the first 21 bits
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

__device__ __host__
inline uint64_t mortonEncode(unsigned int x, unsigned int y, unsigned int z){
    uint64_t answer = 0;
    answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
    return answer;
}

// Assumes: x \in [0,1]
__device__ __host__
inline unsigned int toInt(float x) {
    return (unsigned int) (x * (1<<BITS_PER_COORD));
}

__device__ __host__ 
inline float scaleValue(float x, float minx, float maxlen) {
    float value = fmin(0.7,fmax(0.0, 0.7*(x-minx)/maxlen));
    return value;
}


__global__
void scalePoints(const float3 *values, uint4 *intValues, int numElements, uint32_t intShift,
      float minx, float miny, float minz, float maxlen) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < numElements) {
            intValues[i].x = toInt(scaleValue(values[i].x, minx, maxlen))+intShift;
            intValues[i].y = toInt(scaleValue(values[i].y, miny, maxlen))+intShift;
            intValues[i].z = toInt(scaleValue(values[i].z, minz, maxlen))+intShift;
            intValues[i].w = i;
        }
}
__global__
void computeMortons(const uint4 *values, uint64_t *mortons, int numData, int numQueries) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (numData + numQueries)) {
        uint64_t morton = mortonEncode(
            values[i].x,
            values[i].y,
            values[i].z
        );
        morton = morton << 1; //Shift all bits up to leave LSD available
        if(i < numData) {
            //Data point
            //unset least significant bit
            morton &= ~1; //0b111111.....11110
        } else {
            //Query point
            //set least significant bit
            morton |= 1; //0b0000....00000001
        }
        mortons[i] = morton;
    }
}
#endif

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
inline float scaleValue(float x, float len, float maxlen) {
    return fmin(fmax(0.0, 0.75*(x + 0.5*len)/maxlen), 0.75);
}


__global__
void scalePoints(const float3 *values, uint32_t *intValues, int numElements, float xlen, float ylen, float zlen, float maxlen) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < numElements) {
            intValues[i*3+0] = toInt(scaleValue(values[i].x, xlen, maxlen));
            intValues[i*3+1] = toInt(scaleValue(values[i].y, ylen, maxlen));
            intValues[i*3+2] = toInt(scaleValue(values[i].z, zlen, maxlen));
        }
}
__global__
void computeMortons(const uint32_t *values, uint64_t *mortons, int numData, int numQueries) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (numData + numQueries)) {
        uint64_t morton = mortonEncode(
            values[i*3+0],
            values[i*3+1],
            values[i*3+2]
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

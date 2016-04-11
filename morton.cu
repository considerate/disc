#include "morton.cuh"

inline void morton3D_Decode_for(const uint64_t m, unsigned int& x, unsigned int& y, unsigned int& z){
    x = 0; y = 0; z = 0;
    unsigned int checkbits = 21;

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
    uint32_t scale = (uint32_t) (1<<BITS_PER_COORD);
    return (unsigned int) (x * scale);
}

__device__ __host__ 
inline float scaleValue(float x, float minx, float maxlen) {
    float value = 0.7*(x-minx)/maxlen;
    return value;
}


__global__
void scalePoints(const float3 *values, uint32_t *indices, uint4 *intValues, int numElements, uint32_t intShift,
      float3pair xpair, float3pair ypair, float3pair zpair) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < numElements) {
            float minx = xpair.first.get()->x;
            float maxx = xpair.second.get()->x;
            float miny = ypair.first.get()->y;
            float maxy = ypair.second.get()->y;
            float minz = zpair.first.get()->z;
            float maxz = zpair.second.get()->z;
            float maxlen = fmax(maxx - minx, fmax(maxy - miny, maxz - minz));
            intValues[i].x = toInt(scaleValue(values[i].x, minx, maxlen))+intShift;
            intValues[i].y = toInt(scaleValue(values[i].y, miny, maxlen))+intShift;
            intValues[i].z = toInt(scaleValue(values[i].z, minz, maxlen))+intShift;
            intValues[i].w = indices[i];
        }
}

__global__
void scalePoints(float3idx *values, float3idx *output, int numElements, float3 mins, float maxlen, bool revert) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < numElements) {
            float3 value = values[i].value;
            uint32_t idx = values[i].i;
            if(revert) {
                float x = value.x * maxlen/0.7 + mins.x;
                float y = value.y * maxlen/0.7 + mins.y;
                float z = value.z * maxlen/0.7 + mins.z;
                float3idx result = {{x,y,z}, idx};
                output[i] = result;
            } else {
                float x = 0.7*(value.x - mins.x)/maxlen;
                float y = 0.7*(value.y - mins.y)/maxlen;
                float z = 0.7*(value.z - mins.z)/maxlen;
                float3idx result = {{x,y,z}, idx};
                output[i] = result;
            }
        }
}

__global__
void scalePointsOld(float3 *values, uint4 *intValues, int numElements, uint32_t intShift,
      float3 mins, float maxlen, bool revert) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < numElements) {
            if(revert) {
                float x = values[i].x * maxlen/0.7 + mins.x;
                float y = values[i].y * maxlen/0.7 + mins.y;
                float z = values[i].z * maxlen/0.7 + mins.z;
                values[i].x = x;
                values[i].y = y;
                values[i].z = z;
            } else {
                float x = scaleValue(values[i].x, mins.x, maxlen);
                float y = scaleValue(values[i].y, mins.y, maxlen);
                float z = scaleValue(values[i].z, mins.z, maxlen);
                intValues[i].x = toInt(x)+intShift;
                intValues[i].y = toInt(y)+intShift;
                intValues[i].z = toInt(z)+intShift;
                intValues[i].w = i;
                values[i].x = x;
                values[i].y = y;
                values[i].z = z;
            }
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

__global__
void computeMortons(float3idx *values, uint32_t intShift, uint64_t *mortons, int numData, int numQueries) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (numData + numQueries)) {
        float3 value = values[i].value;
        uint32_t idx = values[i].i;
        uint32_t x = toInt(value.x)+intShift;
        uint32_t y = toInt(value.y)+intShift;
        uint32_t z = toInt(value.z)+intShift;
        uint64_t morton = mortonEncode(
            x,
            y,
            z
        );
        morton = morton << 1; //Shift all bits up to leave LSD available
        if(idx < numData) {
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

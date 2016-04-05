#ifndef _FLOAT_MATH_H
#define _FLOAT_MATH_H

#include <cuda_runtime.h>
#include <vector_types.h>
#include <stdint.h>


__host__ __device__
inline float dot (float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__
inline float3 cross (float3 a, float3 b) {
    float3 result = {
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    };
    return result;
}

__host__ __device__
inline float3 normalize(float3 a) {
    float scale = 1.0/sqrt(dot(a,a));
    float3 result = {a.x*scale, a.y*scale, a.z*scale};
    return result;
}

#endif

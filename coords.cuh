#ifndef _COORDS_H
#define _COORDS_H

#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <stdint.h>
#include <math.h>

typedef struct {
    float3 x, y, z;
} CoordinateSystem;

inline CoordinateSystem unitSystem() {
    CoordinateSystem unit = {{1,0,0},{0,1,0},{0,0,1}};
    return unit;
};

void coordSpaceMatrix(CoordinateSystem from, CoordinateSystem to, float *matrix);
void moveToCoordSpace(CoordinateSystem from, CoordinateSystem to, float3 *values, uint32_t numValues, float3 *result);

#endif

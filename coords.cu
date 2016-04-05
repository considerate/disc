#include "coords.cuh"
#include "float3math.cuh"
#include <stdio.h>

__device__
void calculateTransformMatrix(CoordinateSystem from, CoordinateSystem to, float *matrix) {
    float3 u = to.x;
    float3 v = to.y;
    float3 w = to.z;
    float3 ts[4] = {from.x, from.y, from.z, {0,0,0}};
    for (int i = 0; i < 4; i++) {
        float3 t = ts[i];
        float d0 = dot(u, cross(v,w));
        float d1 = dot(t, cross(v,w));
        float d2 = dot(u, cross(t,w));
        float d3 = dot(u, cross(v,t));
        float e1 = d1/d0;
        float e2 = d2/d0;
        float e3 = d3/d0;
        matrix[i*4+0] = e1;
        matrix[i*4+1] = e2;
        matrix[i*4+2] = e3;
        if(i == 3) {
            matrix[i*4+3] = 1;
        } else {
            matrix[i*4+3] = 0;
        }
    }
}

__device__
float3 multiply4x4x3(float *matrix, float3 a) {
    float u = a.x * matrix[0] + a.y * matrix[4] + a.z * matrix[8] + matrix[12];
    float v = a.x * matrix[1] + a.y * matrix[5] + a.z * matrix[9] + matrix[13];
    float w = a.x * matrix[2] + a.y * matrix[6] + a.z * matrix[10] + matrix[14];
    float3 result = {u,v,w};
    return result;
}

__global__
void transformPoints(float3 *values, uint32_t numValues, float *matrix,  float3 *result) {
    __shared__ float m[16];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < 16) {
        m[i] = matrix[i];
    } 
    __syncthreads();
    if(i < numValues) {
        result[i] = multiply4x4x3(m, values[i]);
    }
}

__global__
void createMatrix(CoordinateSystem from, CoordinateSystem to, float *matrix) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < 1) {
        calculateTransformMatrix(from, to, matrix);   
    }
}

void coordSpaceMatrix(CoordinateSystem from, CoordinateSystem to, float *matrix) {
    createMatrix<<<1,1>>>(from, to, matrix);
}

void moveToCoordSpace(CoordinateSystem from, CoordinateSystem to, float3 *values, uint32_t numValues, float3 *result) {
    float *matrix = NULL;
    size_t matrixSize = 4*4*sizeof(float);
    cudaMalloc((void**) &matrix, matrixSize);
    coordSpaceMatrix(from, to, matrix);
    int threadsPerBlock = 256;
    int blocksPerGrid = (numValues + threadsPerBlock - 1) / threadsPerBlock;
    transformPoints<<<blocksPerGrid, threadsPerBlock>>>(values, numValues, matrix, result);
}

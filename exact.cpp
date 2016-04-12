#include "exact.h"


#define EPSILON 0.0001

typedef struct {
    float x,y,z;
} float3;

typedef struct {
    float3 x, y, z;
} CoordinateSystem;


inline float3 mult (float3 a, float3 b) {
    float3 result = {
        a.x*b.x,
        a.y*b.y,
        a.z*b.z
    };
    return result;
}

inline float dot (float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline float3 cross (float3 a, float3 b) {
    float3 result = {
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    };
    return result;
}

inline float3 normalize(float3 a) {
    float scale = 1.0/sqrt(dot(a,a));
    float3 result = {a.x*scale, a.y*scale, a.z*scale};
    return result;
}

void calculateTransformMatrix(CoordinateSystem from, CoordinateSystem to, float *matrix) {
    float3 u = to.x;
    float3 v = to.y;
    float3 w = to.z;
    for (int i = 0; i < 4; i++) {
        float3 t;
        if(i == 0) {
            t = from.x;
        } else if(i == 1) {
            t = from.y;
        } else if(i == 2) {
            t = from.z;
        } else {
            float3 tmp = {0,0,0};
            t = tmp;
        }
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

inline uint32_t binarySearch(uint64_t *values, uint64_t input, uint32_t len) {
    int32_t imin = 0;
    int32_t imax = len-1;
    while (imin <= imax) {
        uint32_t imid = imin + (imax - imin)/2;
        if (input < values[imid]) {
            imax = imid - 1;
        }
        else {
            imin = imid + 1;
        }
    }
    return (uint32_t) imin;
}
const CoordinateSystem unit = {{1,0,0}, {0,1,0}, {0,0,1}};

float3 findArbitraryTangent(float3 normal) {
    if(fabs(normal.z) > EPSILON) {
        float x = 1.0;
        float y = 1.0;
        float z = (normal.x + normal.y)/normal.z;
        float3 result = {x,y,z};
        return result;
    } else if(fabs(normal.y) > EPSILON) {
        float x = 1.0;
        float z = 1.0;
        float y = (normal.x + normal.z)/normal.y;
        float3 result = {x,y,z};
        return result;
    } else {
        float y = 1.0;
        float z = 1.0;
        float x = (normal.z + normal.y)/normal.x;
        float3 result = {x,y,z};
        return result;
    }
}

const float3 tangentScaling = {0.5, 0.5, 0.5};
const float3 normalScaling = {2.0, 2.0, 2.0};

float3 multiply4x4x3(float *matrix, float3 a) {
    float u = a.x * matrix[0] + a.y * matrix[4] + a.z * matrix[8] + matrix[12];
    float v = a.x * matrix[1] + a.y * matrix[5] + a.z * matrix[9] + matrix[13];
    float w = a.x * matrix[2] + a.y * matrix[6] + a.z * matrix[10] + matrix[14];
    float3 result = {u,v,w};
    return result;
}

float distance (float3 a, float3 b) {
    float3 diff = {
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    };
    return sqrt(dot(diff,diff));
}

void knn(int numData, int numQueries, int k, float3 *values, float3 *normals, uint64_t *nearest) {
    float matrix[16];
    for(int q = 0; q < numQueries; q++) {
        float3 normal = normals[q];
        float3 tangent0 = normalize(findArbitraryTangent(normal));
        float3 tangent1 = cross(normal, tangent0);
        CoordinateSystem querySpace = {
            mult(tangent0, tangentScaling),
            mult(normal, normalScaling),
            mult(tangent1, tangentScaling)
        };
        calculateTransformMatrix(unit, querySpace, matrix);
        float3 query = values[q+numData];
        query = multiply4x4x3(matrix, query);
        for(int d = 0; d < numData; d++) {
            float3 v = values[d];
            v = multiply4x4x3(matrix, v);
            float dist = distance(query, v);
            uint64_t intdist = (dist * (1 << 27));
            uint64_t candidate = (intdist << 32) | d;
            if(d < k) {
                uint32_t loc = binarySearch(&nearest[q*k], candidate, d);
                for(int a = d; a > loc; a--) {
                    nearest[q*k+a] = nearest[q*k+a-1];
                }
                nearest[q*k+loc] = candidate;
            } else if(candidate < nearest[q*k+k-1]) {
                uint32_t loc = binarySearch(&nearest[q*k], candidate, k);
                if(loc == k) {
                    continue;
                }
                for(int a = k-1; a > loc; a--) {
                    nearest[q*k+a] = nearest[q*k+a-1];
                }
                nearest[q*k+loc] = candidate;
            }
        }
    }
}

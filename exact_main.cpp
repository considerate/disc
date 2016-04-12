#include "exact.h"

void initValues(float3 *values, float3 *querynormals, int numElements, int numQueries) {
    float randMax = (float) RAND_MAX;
    for (int i = 0; i < numQueries; ++i) {
        querynormals[i].x = rand()/randMax;
        querynormals[i].y = rand()/randMax;
        querynormals[i].z = rand()/randMax;
        querynormals[i] = normalize(querynormals[i]);
    }
    for (int i = 0; i < numElements; ++i) {
        values[i].x = rand()/randMax;
        values[i].y = rand()/randMax;
        values[i].z = rand()/randMax;
    }
}

int main() {
    srand(0);
    uint32_t numData = 1 << 20;
    uint32_t numQueries = 1 << 3;
    uint32_t numElements = numData + numQueries;
    int k = 4;
    float3 *values = (float3 *) malloc(numElements * sizeof(float3));
    float3 *normals = (float3 *) malloc(numQueries * sizeof(float3));
    uint64_t *nearest = (uint64_t *) malloc(numQueries * k * sizeof(uint64_t));
    initValues(values, normals, numElements, numQueries);
    knn(numData, numQueries, k, values, normals, nearest);
    float matrix[16];
    for(int q = 0; q < numQueries; q++) {
        fprintf(stderr, "%u ", q);
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
        for(int j = 1; j < k; j++) {
            uint32_t d0 = (uint32_t) nearest[q*k+j-1];
            uint32_t d1 = (uint32_t) nearest[q*k+j];
            float3 dv0 = multiply4x4x3(matrix, values[d0]);
            float3 dv1 = multiply4x4x3(matrix, values[d1]);
            float dist0 = distance(query, dv0);
            float dist1 = distance(query, dv1);
            fprintf(stderr, "%f,%f ", dist0, dist1 );
            assert(dist0 <= dist1);
            //fprintf(stderr, "%u ", (uint32_t) (nearest[q*k+j]));
        }
        fprintf(stderr, "\n");
    }
    return 0;
}



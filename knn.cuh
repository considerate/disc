#ifndef _KNN_H
#define _KNN_H
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <limits.h>
#include "morton.cuh"
#ifndef EPSILON
#define EPSILON (0.0001)
#endif
#define ASCENDING (1)
#define DESCENDING (0)
using namespace std;

int nearestNeighbors(int numData, int numQueries, uint32_t k, float3 *values, uint64_t *nearest, float minx, float miny, float minz, float maxlen);
void calculateBounds(float3 *values, int numData, int numQuery, int dataElems, int end, float &minx, float &miny, float &minz, float &maxlen);
int readCSV(const char *filename, float3 *values, int start, int end);
void initValues(float3 *values, float &minx, float &miny, float &minz, float &maxlen, int numElements);

#endif
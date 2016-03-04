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
#include "morton.cu"
#ifndef EPSILON
#define EPSILON (0.0001)
#endif
#define ASCENDING (1)
#define DESCENDING (0)
using namespace std;


void handleError(cudaError_t err, int line) {
    if (err != cudaSuccess) {
        printf("Cuda Error %d", line);
        exit(EXIT_FAILURE);
    }
}

    template<typename T>
string convert_to_binary_string(const T value, bool skip_leading_zeroes = false)
{
    string str;
    bool found_first_one = false;
    const int bits = sizeof(T) * 8;  // Number of bits in the type

    for (int current_bit = bits - 1; current_bit >= 0; current_bit--)
    {
        if ((value & (1ULL << current_bit)) != 0)
        {
            if (!found_first_one)
                found_first_one = true;
            str += '1';
        }
        else
        {
            if (!skip_leading_zeroes || found_first_one)
                str += '0';
        }
    }

    return str;
}

__device__ __host__ inline
uint32_t binarySearch(uint64_t *values, uint64_t input, uint32_t len) {
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

__global__
void compactPoints(uint4 *values, uint64_t *mortons, uint32_t *prefixQueryIndex, uint4 *data, uint64_t *queryIndices, int numData, int numElements) {
    uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numElements) {
        int isQuery = mortons[i] & 1; //check LSD
        if(isQuery) {
            uint32_t nthQ = prefixQueryIndex[i];
            uint64_t qi = i-nthQ+1;
            uint32_t queryindex = values[i].w - numData;
            queryIndices[queryindex] = (qi << 32) | i;
        } else {
            uint32_t numQueriesToLeft = prefixQueryIndex[i];
            uint32_t di = i-numQueriesToLeft;
            data[di] = values[i];
        }
    }
}

__device__  
uint32_t intSqrt(int64_t remainder) {  
    uint64_t place = (uint64_t) 1 << (sizeof (uint64_t) * 8 - 2); // calculated by precompiler = same runtime as: place = 0x40000000  
    while (place > remainder) {
        place /= 4; // optimized by complier as place >>= 2  
    }

    uint64_t root = 0;  
    while (place) {  
        if (remainder >= root+place) {  
            remainder -= root+place;  
            root += place * 2;  
        }  
        root /= 2;  
        place /= 4;  
    }  
    return (uint32_t) root;  
}  

__device__ 
uint64_t distanceSq(uint4 a, uint4 b) {
    int64_t x = (int64_t) b.x - (int64_t) a.x;
    int64_t y = (int64_t) b.y - (int64_t) a.y;
    int64_t z = (int64_t) b.z - (int64_t) a.z;
    int64_t n = (x*x) + (y*y) + (z*z);
    return (uint64_t) intSqrt(n);
}

__device__ inline
void Comparator(uint64_t &a, uint64_t &b, uint32_t dir) {
    uint64_t t;
    if((a > b) == dir) {
        //Swap if a > b;
        t = a;
        a = b;
        b = t;
    }
}


    __device__ 
void bitonicSort(uint64_t *values, uint32_t length,  uint32_t i)
{
    const uint32_t dir = ASCENDING;
    for (uint32_t size = 2; size < length; size <<= 1) {
        //Bitonic merge
        uint32_t ddd = dir ^ ((i & (size / 2)) != 0);

        for (uint32_t stride = (length >> 1); stride > 0; stride >>= 1)
        {
            __syncthreads();
            uint32_t pos = 2 * i - (i & (stride - 1));
            Comparator(
                    values[pos +      0],
                    values[pos + stride],
                    ddd
                    );
        }
    }

    for (uint32_t stride = (length >> 1); stride > 0; stride >>= 1) {
        __syncthreads();
        uint32_t pos = 2 * i - (i & (stride - 1));
        Comparator(
                values[pos +      0],
                values[pos + stride],
                dir
                );
    }
}


__global__
void approxNearest(uint64_t *queryIndices, uint4 *values, uint4 *data, uint64_t *currentNearest, uint32_t k, int numQueries, int numData) {
    extern __shared__ uint64_t candidates[];
    uint32_t q = (uint32_t) blockIdx.x;
    if(q < numQueries) {
        uint64_t iq_both  = queryIndices[q]; // data point right to query point
        uint32_t iq = (uint32_t) (iq_both >> 32);
        uint32_t index = (uint32_t) iq_both;
        uint4 querypoint = values[index];
        uint32_t i = (uint32_t) threadIdx.x;
        if(i < k) {
            iq = min(numData-k-1, max(k-1, iq));
            uint32_t leftidx = iq-i;
            uint32_t rightidx = iq+i+1;
            uint4 left = data[leftidx];
            uint4 right = data[rightidx];
            uint64_t leftdist = distanceSq(querypoint, left);          
            uint64_t rightdist = distanceSq(querypoint, right);          
            printf("DIST: %u => %lu, %lu\n", q, leftdist, rightdist);

            candidates[2*i] = (leftdist << 32) | left.w;
            candidates[2*i+1] = (rightdist << 32) | right.w;
            __syncthreads();
            bitonicSort(candidates, 2*k, 2*i);
            __syncthreads();
            bitonicSort(candidates, 2*k, 2*i+1);
            __syncthreads();
            printf("DIST: (%u,%u) => %u\n", q, i, candidates[i] >> 32);
        }
        //Write to global memory
        if(i < k) {
            currentNearest[q*k+i] = candidates[i];
        }
    }
}

void findCandidates(uint64_t *queryIndices, uint4 *values, uint4 *data, uint64_t *nearest, const uint32_t k, int numQueries, int numData) {
    int threadsPerBlock = k;
    int blocksPerGrid = numQueries;
    approxNearest<<<blocksPerGrid, threadsPerBlock, 2*k*sizeof(uint64_t)>>>(queryIndices, values, data, nearest, k, numQueries, numData);

    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    handleError(err, __LINE__);
}

int getMortons(float3 *devValues, uint4 *devIntValues, uint64_t *devMortons,
        uint32_t shift,
        const float minx, const float miny, const float minz, const float maxlen,
        const int numElements, const int numData, const int numQueries) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    scalePoints<<<blocksPerGrid, threadsPerBlock>>>(devValues, devIntValues, numElements, shift, minx, miny, minz, maxlen);
    computeMortons<<<blocksPerGrid, threadsPerBlock>>>(devIntValues, devMortons, numData, numQueries);

    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    handleError(err, __LINE__);
    return EXIT_SUCCESS;
}

int pointCompaction(uint4 *devIntValues, uint64_t *devMortons, uint32_t *devPrefixQueryIndex, uint4 *devData, uint64_t *devQueryIndices, int numData, int numElements) {
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    compactPoints<<<blocksPerGrid, threadsPerBlock>>>(devIntValues, devMortons, devPrefixQueryIndex, devData, devQueryIndices, numData, numElements);
    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    handleError(err, __LINE__);
    return EXIT_SUCCESS;
}

__global__
void prefixList(uint4 *values, uint32_t *prefix, int numData,  int numElements) {
    uint32_t i = (uint32_t) blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numElements) {
        if(values[i].w < numData) {
            prefix[i] = 0;
        } else {
            prefix[i] = 1;
        }
    }
}

int createPrefixList(uint4 *devIntValues, uint32_t *devPrefixQueryIndex, int numData, int numElements) {
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    prefixList<<<blocksPerGrid, threadsPerBlock>>>(devIntValues, devPrefixQueryIndex, numData, numElements);

    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    handleError(err, __LINE__);

    // in-place prefix sum on list
    thrust::inclusive_scan(thrust::device, devPrefixQueryIndex, devPrefixQueryIndex + numElements, devPrefixQueryIndex);
    err = cudaGetLastError();
    handleError(err, __LINE__);

    return EXIT_SUCCESS;
}


__global__
void mergeNearest(uint64_t *nearest, uint4 *values, uint4 *data, uint64_t *queryIndices, const uint32_t k, int numQueries, int numData) {
    extern __shared__ uint64_t shared[];
    uint64_t *currentNN = (uint64_t *) shared;
    uint32_t *counter = (uint32_t *) &currentNN[k];
    uint64_t *candidates = (uint64_t *) &counter[2*k];
    uint64_t *updatedNN = (uint64_t *) &candidates[2*k];
    uint32_t i = (uint32_t) threadIdx.x;
    uint32_t q = (uint32_t) blockIdx.x;

    if(i < k && q < numQueries) {
        currentNN[i] = nearest[q*k+i];
    }

    if(i < 2*k) {
        counter[i] = i & 1; //1 if odd
    }
    __syncthreads();

    uint32_t offset;
    uint64_t candidate;
    uint32_t loc;
    bool active = true;
    if(q < numQueries) {
        if(i < 2*k) {
            uint64_t iq_both  = queryIndices[q]; // data point right to query point
            uint32_t iq = iq_both >> 32;
            uint32_t index = (uint32_t) iq_both;
            uint4 querypoint = values[index];
            iq = min(numData-k-1, max(k-1, iq));
            bool odd = (i & 1);
            uint32_t idx;
            if(odd) {
                idx = iq+(i/2)+1;
            } else {
                idx = iq-(i/2);
            }
            uint4 datapoint = data[idx];
            uint64_t dist = distanceSq(querypoint, datapoint);          
            candidate = (dist << 32) | datapoint.w;
            loc = binarySearch(currentNN, candidate, k);
            //printf("(%u,%u,%u): %lu = %u\n", q, i, datapoint.w, dist,loc);

            if(loc == k) {
                active = false;
            } else {
                if(loc == 0) {
                    offset = atomicAdd(&counter[0], 1);
                } else {
                    uint64_t current = currentNN[loc-1];
                    if(current != candidate) {
                        offset = atomicAdd(&counter[loc*2], 1);
                    } else {
                        active = false;
                    }
                }
            }
            //printf("(%u,%u): %u \n", q, i, offset);
        }
    }
    __syncthreads();

    // Do a block-wise parallel exclusive prefix sum over counter
    thrust::exclusive_scan(thrust::device, counter, counter + (2*k), counter);

    __syncthreads();

    // for all current nearest
    if(i < k) {
        uint32_t index = counter[2*i + 1];
        if(index < k) {
            updatedNN[index] = currentNN[i];
        }
    }
    __syncthreads();

    // for all new candidate nearest
    if(i < 2*k && active == true) {
        uint32_t index = counter[2*loc] + offset;
        if(index < k) {
            updatedNN[index] = candidate;
        }
    }
    __syncthreads();

    if(i < k && q < numQueries) {
        nearest[q*k+i] = updatedNN[i];
    }
    __syncthreads();
}


__global__ 
void sortMerged(uint64_t *nearest, uint32_t k, uint32_t numQueries) {
    extern __shared__ uint64_t toSort[];
    uint32_t i = (uint32_t) threadIdx.x;
    uint32_t q = (uint32_t) blockIdx.x;
    uint32_t block = (uint32_t) blockIdx.x * blockDim.x;
    if(i < k && q < numQueries) {
        toSort[i] = nearest[block+i];
        __syncthreads();
        bitonicSort(toSort, k, i);
        __syncthreads();
        nearest[block+i] = toSort[i];
    }
}

void mergeStep(uint64_t *nearest, uint4 *values, uint4 *data, uint64_t *queryIndices, const uint32_t k, int numQueries, int numData) {
    int threadsPerBlock = 2*k;
    int blocksPerGrid = numQueries;
    size_t sharedMemorySize = k*sizeof(uint64_t)+ 2*k*sizeof(uint32_t) + 2*k*sizeof(uint64_t) + k*sizeof(uint64_t);
    mergeNearest<<< blocksPerGrid, threadsPerBlock, sharedMemorySize >>>(nearest, values, data, queryIndices, k, numQueries, numData);

    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    handleError(err, __LINE__);
    // nearest now unsorted - sort each block (data points for a query) by distance from q
    sortMerged<<< blocksPerGrid, k, k*sizeof(uint64_t) >>>(nearest, k, numQueries);
    err = cudaGetLastError();
    handleError(err, __LINE__);
}


void initValues(float3 *values, float &minx, float &miny, float &minz, float &maxlen, int numElements) {
    float randMax = (float) RAND_MAX;
    minx = FLT_MAX;
    miny = FLT_MAX;
    minz = FLT_MAX;

    float maxx = FLT_MIN;
    float maxy = FLT_MIN;
    float maxz = FLT_MIN;
    for (int i = 0; i < numElements; ++i) {
        values[i].x = rand()/randMax;
        if(values[i].x < minx) {
            minx = values[i].x;
        }
        if(values[i].x > maxx) {
            maxx = values[i].x;
        }
        values[i].y = rand()/randMax;
        if(values[i].y < miny) {
            miny = values[i].y;
        }
        if(values[i].y > maxy) {
            maxy = values[i].y;
        }
        values[i].z = rand()/randMax;
        if(values[i].z < minz) {
            minz = values[i].z;
        }
        if(values[i].z > maxz) {
            maxz = values[i].z;
        }
    }
    float xl, yl, zl;
    xl = maxx-minx;
    yl = maxy-miny;
    zl = maxz-minz;
    maxlen = fmax(xl,fmax(yl, zl));
}

int main(void)
{
    int numData = 1 << 5;
    int numQueries = 4;
    uint32_t k = 10;
    int numElements = numData + numQueries;

    assert(2*k <= numData);

    //srand(time(NULL));

    size_t valueSize = numElements * sizeof(float3);
    float3 *values = (float3 *) malloc(valueSize);

    size_t intSize = numElements * sizeof(uint4);
    uint4 *intValues = (uint4 *) malloc(intSize);

    size_t dataSize = numData * sizeof(uint4);
    uint4 *data = (uint4 *) malloc(dataSize);

    size_t qiSize = numQueries * sizeof(uint64_t);
    uint64_t *queryIndices = (uint64_t *) malloc(qiSize);

    size_t nearestSize = numQueries * k * sizeof(uint64_t);
    uint64_t *nearest = (uint64_t *) malloc(nearestSize);

    size_t mortonSize = numElements * sizeof(uint64_t);
    uint64_t *mortons = (uint64_t *) malloc(mortonSize);

    size_t prefixSize = numElements * sizeof(uint32_t);

    cudaError_t err = cudaSuccess;

    float3 *devValues = NULL;
    err = cudaMalloc((void **) &devValues, valueSize);
    handleError(err, __LINE__);

    uint4 *devIntValues = NULL;
    err = cudaMalloc((void **) &devIntValues, intSize);
    handleError(err, __LINE__);

    uint64_t *devMortons = NULL;
    err = cudaMalloc((void **) &devMortons, mortonSize);
    handleError(err, __LINE__);

    uint32_t *devPrefixQueryIndex = NULL;
    err = cudaMalloc((void **) &devPrefixQueryIndex, prefixSize);
    handleError(err, __LINE__);

    uint64_t *devQueryIndices = NULL;
    err = cudaMalloc((void **) &devQueryIndices, qiSize);
    handleError(err, __LINE__);

    uint4 *devData = NULL;
    err = cudaMalloc((void **) &devData, dataSize);
    handleError(err, __LINE__);

    uint64_t *devNearest = NULL;
    err = cudaMalloc((void **) &devNearest, nearestSize);
    handleError(err, __LINE__);

    float minx, miny, minz, maxlen;
    initValues(values, minx, miny, minz, maxlen, numElements);
    printf("lens: (%f,%f,%f,%f)\n", minx, miny, minz, maxlen);

    err = cudaMemcpy(devValues, values, valueSize, cudaMemcpyHostToDevice);
    handleError(err, __LINE__);

    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);

    for(int j = 0; j < 5; ++j) {
        printf("Iteration: %d\n",j);
        float shift = j*0.05;
        uint32_t intShift = (uint32_t) (shift * (1 << 21));
        getMortons(devValues, devIntValues, devMortons,
                intShift,
                minx, miny, minz, maxlen,
                numElements, numData, numQueries);

        err = cudaMemcpy(intValues, devIntValues, intSize, cudaMemcpyDeviceToHost);
        handleError(err, __LINE__);

        //sort values in morton code order
        thrust::sort_by_key(thrust::device, devMortons, devMortons + numElements, devIntValues);

        createPrefixList(devIntValues, devPrefixQueryIndex, numData, numElements);
        err = cudaMemcpy(intValues, devIntValues, intSize, cudaMemcpyDeviceToHost);
        handleError(err, __LINE__);
        pointCompaction(devIntValues, devMortons, devPrefixQueryIndex, devData, devQueryIndices, numData, numElements);
        err = cudaMemcpy(queryIndices, devQueryIndices, qiSize, cudaMemcpyDeviceToHost);
        handleError(err, __LINE__);

        for(int a = 0; a < numElements; ++a) {
          uint4 point = intValues[a];
          if(point.w < numData) {
            printf("D(%u,%u,%u,%u)\n", point.x, point.y, point.z, point.w);
          } else {
            printf("Q(%u,%u,%u,%u)\n", point.x, point.y, point.z, point.w);
          }
        }

        if(j == 0) {
            findCandidates(devQueryIndices, devIntValues, devData, devNearest, k, numQueries, numData);
        } else {
            mergeStep(devNearest, devIntValues, devData, devQueryIndices, k, numQueries, numData);
        }
        err = cudaMemcpy(nearest, devNearest, nearestSize, cudaMemcpyDeviceToHost);
        handleError(err, __LINE__);
        for(int a = 0; a < numQueries; ++a) {
            printf("[");
            for(int b = 0; b < k; ++b) {
                printf("(%u,%u),", (uint32_t) (nearest[a*k + b] >> 32), (uint32_t) (nearest[a*k + b]) );
            }
            printf("]\n");
        }
    }

    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);

    err = cudaMemcpy(nearest, devNearest, nearestSize, cudaMemcpyDeviceToHost);
    handleError(err, __LINE__);
    err = cudaMemcpy(data, devData, dataSize, cudaMemcpyDeviceToHost);
    handleError(err, __LINE__);

    printf("Time elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

    err = cudaFree(devValues);
    handleError(err, __LINE__);

    err = cudaFree(devMortons);
    handleError(err, __LINE__);

    // Free host memory
    free(values);
    free(mortons);

    err = cudaDeviceReset();
    handleError(err, __LINE__);

    return EXIT_SUCCESS;
}

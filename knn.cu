#include "knn.cuh"
#include "morton.cuh"
#include "coords.cuh"
#include "float3math.cuh"
#define PI 3.145927

__host__ __device__
uint4 toUint4(float3 value, uint32_t index) {
    uint32_t scale = (1 << 30);
    uint4 result = {value.x * scale, value.y * scale, value.z * scale, index};
    return result;
}

__device__
void calculateTransformMatrixKNN(CoordinateSystem from, CoordinateSystem to, float *matrix) {
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

__host__ __device__
float3 multiply4x4x3KNN(float *matrix, float3 a) {
    float u = a.x * matrix[0] + a.y * matrix[4] + a.z * matrix[8] + matrix[12];
    float v = a.x * matrix[1] + a.y * matrix[5] + a.z * matrix[9] + matrix[13];
    float w = a.x * matrix[2] + a.y * matrix[6] + a.z * matrix[10] + matrix[14];
    float3 result = {u,v,w};
    return result;
}

__host__ __device__
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

void handleError(cudaError_t err, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Cuda Error %d", line);
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
void compactPoints(float3idx *values, uint64_t *mortons, uint32_t *prefixQueryIndex, uint32_t *reverseIndices, float3idx *data, uint64_t *queryIndices, int numData, int numElements) {
    uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numElements) {
        int isQuery = mortons[i] & 1; //check LSD
        if(isQuery) {
            uint32_t nthQ = prefixQueryIndex[i];
            uint64_t qi = i-nthQ+1;
            uint32_t rev = reverseIndices[values[i].i];
            uint32_t queryindex = rev - numData;
            //printf("%d -> (%d - %d) = %d \n", values[i].i, rev, numData, queryindex);
            queryIndices[queryindex] = (qi << 32) | i;
        } else {
            uint32_t numQueriesToLeft = prefixQueryIndex[i];
            uint32_t di = i-numQueriesToLeft;
            data[di] = values[i];
        }
    }
}

__global__
void compactPointsOld(uint4 *values, uint64_t *mortons, uint32_t *prefixQueryIndex, uint4 *data, uint64_t *queryIndices, int numData, int numElements) {
    uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numElements) {
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

__device__
float fdist(float3 a, float3 b) {
    float3 sub = {b.x - a.x, b.y - a.y, b.z - a.z};
    return sqrt(dot(sub,sub));
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

        for (uint32_t stride = size / 2; stride > 0; stride >>= 1)
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

    for (uint32_t stride = length / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        uint32_t pos = 2 * i - (i & (stride - 1));
        Comparator(
                values[pos +      0],
                values[pos + stride],
                dir
                );
    }
}


__device__ inline void swap(uint64_t & a, uint64_t & b)
{
    register uint64_t tmp = a;
    a = b;
    b = tmp;
}


__device__
void bitonicSort2(uint64_t *values, uint32_t size, uint32_t tid) {

    // Parallel bitonic sort.
    for (int k = 2; k <= size; k *= 2) {
        // Bitonic merge:
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = tid ^ j;

            if (ixj > tid) {
                if ((tid & k) == 0) {
                    if (values[tid] > values[ixj]) {
                        swap(values[tid], values[ixj]);
                    }
                }
                else {
                    if (values[tid] < values[ixj]) {
                        swap(values[tid], values[ixj]);
                    }
                }
            }

            __syncthreads();
        }
    }
}
__global__
void approxNearest(uint64_t *queryIndices, uint4 *values, float3 *floatvalues, uint4 *data, uint64_t *currentNearest, uint32_t k, uint32_t lambdak, int numQueries, int numData) {
    extern __shared__ uint64_t candidates[];
    uint32_t q = (uint32_t) blockIdx.x;
    uint32_t numThreads = (uint32_t) blockDim.x;
    const uint64_t maxDist = (uint64_t) 0xffffffff; 
    if(q < numQueries) {
        uint64_t iq_both  = queryIndices[q]; // data point right to query point
        uint32_t iq = (uint32_t) (iq_both >> 32);
        uint32_t index = (uint32_t) iq_both;
        uint4 querypoint = values[index];
        float3 queryp = floatvalues[querypoint.w];
        uint32_t i = (uint32_t) threadIdx.x;
        if(i < lambdak) {
            iq = min(numData-lambdak-1, max(lambdak-1, iq));
            uint32_t leftidx = iq-i;
            uint32_t rightidx = iq+i+1;
            uint4 left = data[leftidx];
            uint4 right = data[rightidx];
            
            float3 leftp = floatvalues[left.w];
            float3 rightp = floatvalues[right.w];
            float leftd = fdist(queryp, leftp);
            float rightd = fdist(queryp, rightp);
            uint64_t prec = (uint64_t) (1 << 30);
            uint64_t leftdist = (uint64_t) (leftd * prec);
            uint64_t rightdist = (uint64_t) (rightd * prec);
            
            candidates[2*i] = (uint64_t) leftdist << 32 | left.w;
            candidates[2*i+1] = (uint64_t) rightdist << 32 | right.w;
        } else if(i < numThreads) {
            candidates[2*i] = (maxDist << 32) | 0;
            candidates[2*i+1] = (maxDist << 32) | 0;
        }
        __syncthreads();
        if(i < numThreads) {
            bitonicSort(candidates, 2*numThreads, i);
        }
        //Write to global memory
        if(i < k) {
            //printf("%u %u %u\n", q, i, (uint32_t) (candidates[i] >> 32) );
            currentNearest[q*k+i] = candidates[i];
        }
    }
}


__device__
float3 toFloat3(uint4 value) {
    float scale = 1.0 / (float) (1 << 21);
    float x = value.x * scale;
    float y = value.y * scale;
    float z = value.z * scale;
    float3 result = {x,y,z};
    return result;
}


__host__ __device__
float3 div(float3 a, float3 b) {
    float3 result = {
        a.x / b.x,
        a.y / b.y,
        a.z / b.z
    };
    return result;
}

__host__ __device__
float3 mult(float3 a, float3 b) {
    float3 result = {
        a.x * b.x,
        a.y * b.y,
        a.z * b.z
    };
    return result;
}

__global__
void approxNearestEllipsoid(uint64_t *queryIndices, float3idx *values, float3idx *data, uint64_t *currentNearest, uint32_t k, uint32_t lambdak, int numQueries, int numData,
    float3 normalScaling, float3 tangentScaling,
    CoordinateSystem bucketSpace, float3 querynormals[]) {
    extern __shared__ uint64_t candidates[];
    float *toEllipsoid = (float *) &candidates[2*lambdak];
    uint32_t q = (uint32_t) blockIdx.x;
    uint32_t numThreads = (uint32_t) blockDim.x;
    const uint64_t maxDist = (uint64_t) UINT32_MAX; 
    uint64_t iq_both;
    uint32_t iq,index;
    float3idx querypoint;
    if(q < numQueries) {
        iq_both  = queryIndices[q]; // data point right to query point
        iq = (uint32_t) (iq_both >> 32);
        index = (uint32_t) iq_both;
        querypoint = values[index];
    }
    if(q < numQueries) {
        float3 querynormal = querynormals[querypoint.i - numData];
        float3 tangent0 = normalize(findArbitraryTangent(querynormal));
        float3 tangent1 = cross(querynormal, tangent0);
        uint32_t i = (uint32_t) threadIdx.x;
        if(i < 1) {
            CoordinateSystem ellipsoidSpace = {
                mult(tangent0, tangentScaling), 
                mult(querynormal, normalScaling),
                mult(tangent1, tangentScaling)
            };
            calculateTransformMatrixKNN(bucketSpace, ellipsoidSpace, toEllipsoid);
        }
        __syncthreads();
        if(i < lambdak) {
            iq = min(numData-lambdak-1, max(lambdak-1, iq));
            uint32_t leftidx = iq-i;
            uint32_t rightidx = iq+i+1;
            float3idx left = data[leftidx];
            float3idx right = data[rightidx];
             
            //printf("%u : %u - %u : %u \n", leftidx, rightidx, left.i, right.i);
            // move to ellipsoid coords
            float3 query = multiply4x4x3KNN(toEllipsoid, querypoint.value);
            float3 fleft = multiply4x4x3KNN(toEllipsoid, left.value);
            float3 fright = multiply4x4x3KNN(toEllipsoid, right.value);
            float leftd = fdist(query, fleft);          
            float rightd = fdist(query, fright);          
            //printf("%f %f\n", leftd, rightd);
            uint64_t prec = (uint64_t) (1 << 30);
            uint64_t leftdist = (uint64_t) (leftd * prec);
            uint64_t rightdist = (uint64_t) (rightd * prec);
            candidates[2*i] = (leftdist << 32) | left.i;
            candidates[2*i+1] = (rightdist << 32) | right.i;
        } else if(i < numThreads) {
            candidates[2*i] = (maxDist << 32) | 0xdeadbeef;
            candidates[2*i+1] = (maxDist << 32) | 0xdeadbeef;
        }
        __syncthreads();
        if(i < numThreads) {
            bitonicSort(candidates, 2*numThreads, i);
        }
        __syncthreads();
        //Write to global memory
        if(i < k) {
            uint32_t queryIndex = querypoint.i-numData;
            uint32_t idx = queryIndex*k+i;
            currentNearest[idx] = candidates[i];
        }
    }
}

uint32_t log2(uint32_t x) {
   uint32_t l;
   for(l=0; x>1; x = (x >> 1), l++);
   return l;
} 

void findCandidates(uint64_t *queryIndices, uint4 *values, float3* floatvalues, uint4 *data, uint64_t *nearest, const uint32_t k, int numQueries, int numData, uint32_t lambda) {
    uint32_t lambdak = k*lambda;
    int logn = log2(lambdak - 1);
    int threadsPerBlock = 1 << (logn+1);
    int blocksPerGrid = numQueries;
    approxNearest<<<blocksPerGrid, threadsPerBlock, 2*threadsPerBlock*sizeof(uint64_t)>>>(queryIndices, values, floatvalues, data, nearest, k, lambdak,  numQueries, numData);

    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    handleError(err, __LINE__);
}

void findCandidatesEllipsoid(uint64_t *queryIndices, float3idx *values, float3idx *data, uint64_t *nearest, const uint32_t k, int numQueries, int numData,
    float3 normalScaling, float3 tangentScaling,
    CoordinateSystem bucketSpace, float3 querynormals[], uint32_t lambda) {
    
    uint32_t lambdak = k*lambda;
    int logn = log2(lambdak - 1);
    int threadsPerBlock = 1 << (logn+1);
    int blocksPerGrid = numQueries;
    approxNearestEllipsoid<<<blocksPerGrid, threadsPerBlock, 2*threadsPerBlock*sizeof(uint64_t)+16*sizeof(float)>>>(
        queryIndices, values, data, nearest, k, lambdak, numQueries, numData,
        normalScaling, tangentScaling,
        bucketSpace, querynormals
    );

    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    handleError(err, __LINE__);
}

void scaleValues(float3 *devValues, uint4 *devIntValues,
        uint32_t shift,
        int numElements,
        float3 mins, float maxlen,
        bool revert) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    cudaError_t err = cudaSuccess;
    scalePointsOld<<<blocksPerGrid, threadsPerBlock>>>(devValues, devIntValues, numElements, shift, mins, maxlen, revert);
    err = cudaGetLastError();
    handleError(err, __LINE__);
}

void scaleValues(float3idx *devValues, float3idx *output,
        int numElements,
        float3 mins, float maxlen,
        bool revert) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    cudaError_t err = cudaSuccess;
    scalePoints<<<blocksPerGrid, threadsPerBlock>>>(devValues, output, numElements, mins, maxlen, revert);
    err = cudaGetLastError();
    handleError(err, __LINE__);
}

int getMortonsOld(uint4 *devIntValues, uint64_t *devMortons, const int numData, const int numQueries) {
    int threadsPerBlock = 256;
    int numElements = numData + numQueries;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    cudaError_t err = cudaSuccess;

    computeMortons<<<blocksPerGrid, threadsPerBlock>>>(devIntValues, devMortons, numData, numQueries);
    err = cudaGetLastError();
    handleError(err, __LINE__);
    return EXIT_SUCCESS;
}

int getMortons(float3idx *values, uint32_t shift, uint64_t *devMortons, const int numData, const int numQueries) {
    int threadsPerBlock = 256;
    int numElements = numData + numQueries;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    cudaError_t err = cudaSuccess;
    computeMortons<<<blocksPerGrid, threadsPerBlock>>>(values, shift, devMortons, numData, numQueries);

    err = cudaGetLastError();
    handleError(err, __LINE__);
    return EXIT_SUCCESS;
}

int pointCompaction(float3idx *values, uint64_t *devMortons, uint32_t *devPrefixQueryIndex, uint32_t *reverseIndices, float3idx *devData, uint64_t *devQueryIndices, int numData, int numElements) {
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    compactPoints<<<blocksPerGrid, threadsPerBlock>>>(values, devMortons, devPrefixQueryIndex, reverseIndices, devData, devQueryIndices, numData, numElements);
    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    handleError(err, __LINE__);
    return EXIT_SUCCESS;
}

int pointCompactionOld(uint4 *devIntValues, uint64_t *devMortons, uint32_t *devPrefixQueryIndex, uint4 *devData, uint64_t *devQueryIndices, int numData, int numElements) {
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    compactPointsOld<<<blocksPerGrid, threadsPerBlock>>>(devIntValues, devMortons, devPrefixQueryIndex, devData, devQueryIndices, numData, numElements);
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

__global__
void prefixList(float3idx *values, uint32_t *prefix, int numData,  int numElements) {
    uint32_t i = (uint32_t) blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numElements) {
        if(values[i].i < numData) {
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

int createPrefixList(float3idx *values, uint32_t *devPrefixQueryIndex, int numData, int numElements) {
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    prefixList<<<blocksPerGrid, threadsPerBlock>>>(values, devPrefixQueryIndex, numData, numElements);

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
void mergeNearest(uint64_t *nearest, uint4 *values, float3 *floatvalues, uint4 *data, uint64_t *queryIndices, const uint32_t k, uint32_t lambdak, int numQueries, int numData) {
    extern __shared__ uint64_t shared[];
    uint32_t numThreads = (uint32_t) blockDim.x;
    uint64_t *currentNN = (uint64_t *) shared;
    uint32_t *counter = (uint32_t *) &currentNN[k];
    uint32_t *counter_scan = (uint32_t *) &counter[2*k];
    uint64_t *updatedNN = (uint64_t *) &counter_scan[2*k];
    uint32_t i = (uint32_t) threadIdx.x;
    uint32_t q = (uint32_t) blockIdx.x;

    if(i < k && q < numQueries) {
        currentNN[i] = nearest[q*k+i];
    }

    if(i < 2*k && q < numQueries) {
        counter[i] = i & 1; //1 if odd
    }
    __syncthreads();

    uint32_t offset = 0;
    uint64_t candidate;
    uint32_t loc;
    bool active = true;
    if(q < numQueries) {
        if(i < 2*lambdak) {
            uint64_t iq_both  = queryIndices[q]; // data point right to query point
            uint32_t iq = iq_both >> 32;
            uint32_t index = (uint32_t) iq_both;
            uint4 querypoint = values[index];
            float3 queryp = floatvalues[querypoint.w];
            iq = min(numData-lambdak-1, max(lambdak-1, iq));
            bool odd = (i & 1);
            uint32_t idx;
            if(odd) {
                idx = iq+(i/2)+1;
            } else {
                idx = iq-(i/2);
            }
            uint4 datapoint = data[idx];
            float3 p = floatvalues[datapoint.w];
            uint64_t prec = (uint64_t) (1 << 30);
            float d = fdist(queryp, p);
            uint64_t dist = d * prec;
            //uint64_t dist = distanceSq(querypoint,datspoint);          
            candidate = (dist << 32) | datapoint.w;
            loc = binarySearch(currentNN, candidate, k);

            if(loc == k) {
                active = false;
            } else {
                uint32_t index = (loc != 0) ? (loc-1) : 0;
                uint64_t current = currentNN[index];
                if(current == candidate) {
                    active = false;
                } else {
                    offset = atomicAdd(&counter[loc*2], (uint32_t) 1);
                }
            }
        } else if(i < numThreads) {
            candidate = UINT64_MAX;
            active = false;
            offset = 0;
        }
    }

    // Do a block-wise parallel exclusive prefix sum over counter
    __syncthreads();
    if(i == 0) {
        thrust::exclusive_scan(thrust::device, counter, counter + (2*k), counter_scan);
    }
    __syncthreads();

    // for all current nearest
    if(q < numQueries) {
        if(i < k) {
            uint32_t index = counter_scan[2*i + 1];
            if(index < k) {
                updatedNN[index] = currentNN[i];
            }
        }
        __syncthreads();

        // for all new candidate nearest
        if(i < 2*lambdak && active == true) {
            uint32_t index = counter_scan[2*loc] + offset;
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
}

__global__
void mergeNearestEllipsoid(uint64_t *nearest, float3idx *values,  float3idx *data, uint64_t *queryIndices, const uint32_t k, uint32_t lambdak, int numQueries, int numData,
    float3 normalScaling, float3 tangentScaling,
    CoordinateSystem bucketSpace,  float3 querynormals[], uint32_t intShift) {
    extern __shared__ uint64_t shared[];
    uint32_t numThreads = (uint32_t) blockDim.x;
    uint64_t *currentNN = (uint64_t *) shared;
    uint32_t *counter = (uint32_t *) &currentNN[k];
    uint32_t *counter_scan = (uint32_t *) &counter[2*k];
    uint64_t *updatedNN = (uint64_t *) &counter_scan[2*k];
    float *toEllipsoid = (float *) &updatedNN[k];
    uint32_t i = (uint32_t) threadIdx.x;
    uint32_t q = (uint32_t) blockIdx.x;
    uint64_t iq_both;
    uint32_t iq,index;
    float3idx querypoint;
    if(q < numQueries) {
        iq_both  = queryIndices[q]; // data point right to query point
        iq = (uint32_t) (iq_both >> 32);
        index = (uint32_t) iq_both;
        querypoint = values[index];
    }
    if(i < k && q < numQueries) {
        uint32_t idx = (querypoint.i-numData)*k + i;
        currentNN[i] = nearest[idx];
    }

    if(i < 2*k && q < numQueries) {
        counter[i] = i & 1; //1 if odd
    }
    __syncthreads();

    uint32_t offset = 0;
    uint64_t candidate;
    uint32_t loc;
    bool active = true;
    if(q < numQueries) {
        float3 query = querypoint.value;
        iq = min(numData-lambdak-1, max(lambdak-1, iq));
        float3 querynormal = querynormals[querypoint.i - numData];
        float3 tangent0 = normalize(findArbitraryTangent(querynormal));
        float3 tangent1 = cross(querynormal, tangent0);
        if(i < 1) {
            CoordinateSystem ellipsoidSpace = {
                mult(tangent0, tangentScaling), 
                mult(querynormal, normalScaling),
                mult(tangent1, tangentScaling)
            };
            calculateTransformMatrixKNN(bucketSpace, ellipsoidSpace, toEllipsoid);
        }
        __syncthreads();
        if(i < 2*lambdak) {
            bool odd = (i & 1);
            uint32_t idx;
            if(odd) {
                idx = iq+(i/2)+1;
            } else {
                idx = iq-(i/2);
            }
            float3idx datapoint = data[idx];
            float3 point = datapoint.value;
            // move to ellipsoid coords
            query = multiply4x4x3KNN(toEllipsoid, query);
            point = multiply4x4x3KNN(toEllipsoid, point);
            float d = fdist(query, point);          
            uint64_t prec = (uint64_t) (1 << 30);
            uint64_t dist = d * prec;

            candidate = (dist << 32) | datapoint.i;
            loc = binarySearch(currentNN, candidate, k);

            if(loc == k) {
                active = false;
            } else {
                uint32_t index = (loc != 0) ? (loc-1) : 0;
                uint64_t current = currentNN[index];
                uint64_t lowend = 0xffffffff;
                if((current & lowend) == (candidate & lowend)) {
                    active = false;
                } else {
                    offset = atomicAdd(&counter[loc*2], (uint32_t) 1);
                }
            }
        } else if(i < numThreads) {
            candidate = UINT64_MAX;
            active = false;
            offset = 0;
        }
    }

    // Do a block-wise parallel exclusive prefix sum over counter
    __syncthreads();
    if(i == 0) {
        thrust::exclusive_scan(thrust::device, counter, counter + (2*k), counter_scan);
    }
    __syncthreads();

    // for all current nearest
    if(q < numQueries) {
        if(i < k) {
            uint32_t index = counter_scan[2*i + 1];
            if(index < k) {
                updatedNN[index] = currentNN[i];
            }
        }
        __syncthreads();

        // for all new candidate nearest
        if(i < 2*lambdak && active == true) {
            uint32_t index = counter_scan[2*loc] + offset;
            if(index < k) {
                updatedNN[index] = candidate;
            }
        }
        __syncthreads();

        if(i < k) {
            uint32_t idx = (querypoint.i-numData)*k + i;
            nearest[idx] = updatedNN[i];
        }
        __syncthreads();
    }
}

__global__ 
void sortMerged(uint64_t *nearest, uint32_t k, uint32_t numQueries, uint32_t numData, uint4 *values, uint64_t *queryIndices) {
    extern __shared__ uint64_t toSort[];
    uint32_t i = (uint32_t) threadIdx.x;
    uint32_t q = (uint32_t) blockIdx.x;
    uint32_t numThreads = (uint32_t) blockDim.x;
    uint64_t iq_both;
    uint4 querypoint;
    if(q < numQueries) {
        iq_both  = queryIndices[q]; // data point right to query point
        uint32_t index = (uint32_t) iq_both;
        querypoint = values[index];
    }
    if(i < k && q < numQueries) {
        toSort[i] = nearest[(querypoint.w-numData)*k+i];
        __syncthreads();
    } else if(i < numThreads && q < numQueries) {
        toSort[i] = UINT64_MAX;
        __syncthreads();
    }
    if(i < numThreads && q < numQueries) {
        bitonicSort2(toSort, numThreads, i);
        __syncthreads();
    }

    if(i < k && q < numQueries) {
        nearest[(querypoint.w-numData)*k+i] = toSort[i];
    }

}

__global__ 
void sortMerged(uint64_t *nearest, uint32_t k, uint32_t numQueries, uint32_t numData, float3idx *values, uint64_t *queryIndices) {
    extern __shared__ uint64_t toSort[];
    uint32_t i = (uint32_t) threadIdx.x;
    uint32_t q = (uint32_t) blockIdx.x;
    uint32_t numThreads = (uint32_t) blockDim.x;
    uint64_t iq_both;
    float3idx querypoint;
    if(q < numQueries) {
        iq_both  = queryIndices[q]; // data point right to query point
        uint32_t index = (uint32_t) iq_both;
        querypoint = values[index];
    }
    if(i < k && q < numQueries) {
        toSort[i] = nearest[(querypoint.i-numData)*k+i];
        __syncthreads();
    } else if(i < numThreads && q < numQueries) {
        toSort[i] = UINT64_MAX;
        __syncthreads();
    }
    if(i < numThreads && q < numQueries) {
        bitonicSort2(toSort, numThreads, i);
        __syncthreads();
    }

    if(i < k && q < numQueries) {
        nearest[(querypoint.i-numData)*k+i] = toSort[i];
    }

}

void mergeStep(uint64_t *nearest, uint4 *values, float3 *floatvalues, uint4 *data, uint64_t *queryIndices, const uint32_t k, int numQueries, int numData, uint32_t lambda) {
    uint32_t lambdak = k * lambda;
    uint32_t logn = log2(lambdak - 1);
    int threadsPerBlock = 1 << (logn + 2); // 2*k
    int blocksPerGrid = numQueries;
    size_t sharedMemorySize = k*sizeof(uint64_t) + 2*k*sizeof(uint32_t) + 2*k*sizeof(uint32_t) + k*sizeof(uint64_t);
    mergeNearest<<< blocksPerGrid, threadsPerBlock, sharedMemorySize >>>(nearest, values, floatvalues, data, queryIndices, k, lambdak, numQueries, numData);

    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    handleError(err, __LINE__);

    // nearest now unsorted - sort each block (data points for a query) by distance from q
    uint32_t logn2 = log2(k - 1);
    int sortSize = 1 << (logn2 + 1);
    sortMerged<<< blocksPerGrid, k, sortSize*sizeof(uint64_t) >>>(nearest, k, numQueries, numData, values, queryIndices);
    err = cudaGetLastError();
    handleError(err, __LINE__);
}

void mergeStepEllipsoid(uint64_t *nearest, float3idx *values, float3idx *data, uint64_t *queryIndices, const uint32_t k, int numQueries, int numData,
    float3 normalScaling, float3 tangentScaling,
    CoordinateSystem bucketSpace, float3 querynormals[], uint32_t intShift, uint32_t lambda) {
    uint32_t lambdak = k * lambda;
    uint32_t logn = log2(lambdak - 1);
    int threadsPerBlock = 1 << (logn + 2); // 2*k
    int blocksPerGrid = numQueries;
    size_t sharedMemorySize = k*sizeof(uint64_t) + 2*k*sizeof(uint32_t) + 2*k*sizeof(uint32_t) + k*sizeof(uint64_t) + 16 * sizeof(float);
    mergeNearestEllipsoid<<< blocksPerGrid, threadsPerBlock, sharedMemorySize >>>(nearest, values, data, queryIndices, k, lambdak, numQueries, numData,
    normalScaling, tangentScaling, bucketSpace, querynormals, intShift);

    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    handleError(err, __LINE__);

    // nearest now unsorted - sort each block (data points for a query) by distance from q
    uint32_t logn2 = log2(k - 1);
    int sortSize = 1 << (logn2 + 1);
    sortMerged<<< blocksPerGrid, sortSize, sortSize*sizeof(uint64_t) >>>(nearest, k, numQueries, numData, values, queryIndices);
    err = cudaGetLastError();
    handleError(err, __LINE__);
}

__global__
void copyAllValues(float3 *values, uint32_t numElements, float3 *result) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(i < numElements) {
        result[i] = values[i];
    }
}

struct compare_x {
    __host__ __device__
    bool operator()(float3 a, float3 b)
    {
        return a.x < b.x;
    }
};

struct compare_y {
    __host__ __device__
    bool operator()(float3 a, float3 b)
    {
        return a.y < b.y;
    }
};

struct compare_z {
    __host__ __device__
    bool operator()(float3 a, float3 b)
    {
        return a.z < b.z;
    }
};

struct compare_x_idx {
    __host__ __device__
    bool operator()(float3idx a, float3idx b)
    {
        return a.value.x < b.value.x;
    }
};

struct compare_y_idx {
    __host__ __device__
    bool operator()(float3idx a, float3idx b)
    {
        return a.value.y < b.value.y;
    }
};

struct compare_z_idx {
    __host__ __device__
    bool operator()(float3idx a, float3idx b)
    {
        return a.value.z < b.value.z;
    }
};


__global__
void markBucketKernel(int numData, int numQueries, float3 *querynormals, float3* bucketnormals, int buckets, int bucket, uint32_t *marks) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(i < numData) {
        marks[i] = 0;
    } else if(i < (numData + numQueries)) {
        float3 normal = querynormals[i-numData];
        float maxdot = -2.0;
        uint32_t maxbucket = buckets;
        for (uint32_t b = 0; b < buckets; b++) {
            float d = dot(normal, bucketnormals[b]);
            if(d >= maxdot) {
                maxdot = d;
                maxbucket = b;
            }
        }
        if(maxbucket == bucket) {
            marks[i] = 1;
        } else {
            marks[i] = 2;
        }
    }
}

void markInBucket(int numData, int numQueries, float3 *querynormals, float3 *bucketnormals, int buckets, int bucket, uint32_t *marks) {
    int numElements = numData + numQueries;
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    markBucketKernel<<<blocksPerGrid, threadsPerBlock>>>(numData, numQueries, querynormals, bucketnormals, buckets, bucket, marks);
}

__global__
void notInBucketKernel(uint32_t *marks, int numElements) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(i < numElements) {
        if(marks[i] == 2) {
            marks[i] = 1;
        } else {
            marks[i] = 0;
        }
    }
}

void notInBucket(uint32_t *marks, int numElements) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    notInBucketKernel<<<blocksPerGrid, threadsPerBlock>>>(marks, numElements);
}

__global__
void originalIndices(int numElements, float3 *values, float3idx *output) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(i < numElements) {
        float3 value = values[i];
        float3idx result = {value, i};
        output[i] = result;
    }
}

void storeOriginalIndices(int numElements, float3 *values, float3idx *output) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    originalIndices<<<blocksPerGrid, threadsPerBlock>>>(numElements, values, output);
}

__global__
void bucketIndices(uint4 *intValues, int numElements, uint32_t *bucketIndices) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(i < numElements) {
        bucketIndices[intValues[i].w] = i;
    }
}

void storeBucketIndices(uint4 *intValues, int numElements, uint32_t *bucketIdx) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    bucketIndices<<<blocksPerGrid, threadsPerBlock>>>(intValues, numElements, bucketIdx);
}

__global__
void calculateReverseIndices(float3idx *values, int numElements, uint32_t *revIndices) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(i < numElements) {
        revIndices[values[i].i] = i;
    }
}

void reverseIndices(float3idx *values, int numElements, uint32_t *revIndices) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    calculateReverseIndices<<<blocksPerGrid, threadsPerBlock>>>(values, numElements, revIndices);
}

int nearestNeighborsEllipsoid(int numData, int numQueries, uint32_t k, float3 *values, float3 *querynormals, uint64_t *nearest, const uint32_t lambda, const float compressionRate) {
    const int buckets = 18;
    /*
    Create icosahedron (even distribution) and 
    add the octahedron to create one bucket for each axis.

    This results in slightly unveven buckets but points on the
    same surface are likely to be in the same bucket.
    */
    float d = sinf(PI/4.0);
    float3 bucketnormals[buckets] = {
        {0,0,1}, {0,1,0}, {1,0,0},
        {0,0,-1}, {0,-1,0}, {-1,0,0},
        {0, d, d}, {0, d, -d}, {0, -d, d}, {0, -d, -d},
        {d, 0, d}, {d, 0, -d}, {-d, 0, d}, {-d, 0, -d},
        {d, d, 0}, {d, -d, 0}, {-d, d, 0}, {-d, -d, 0}
    };
    size_t bucketNormalSize = buckets * sizeof(float3);

    int numElements = numData + numQueries;

    size_t indexedSize = numElements * sizeof(float3idx);
    size_t dataSize = numData * sizeof(float3idx);
    size_t qiSize = numQueries * sizeof(uint64_t);
    size_t nearestSize = numQueries * k * sizeof(uint64_t);
    size_t queryNormalSize = numQueries * sizeof(float3);
    size_t mortonSize = numElements * sizeof(uint64_t);
    size_t prefixSize = numElements * sizeof(uint32_t);

    cudaError_t err = cudaSuccess;

    size_t valueSize = numElements * sizeof(float3);
    float3 *devValues = NULL;
    err = cudaMalloc((void **) &devValues, valueSize);
    handleError(err, __LINE__);

    uint64_t *devMortons = NULL;
    err = cudaMalloc((void **) &devMortons, mortonSize);
    handleError(err, __LINE__);

    uint32_t *devPrefixQueryIndex = NULL;
    err = cudaMalloc((void **) &devPrefixQueryIndex, prefixSize);
    handleError(err, __LINE__);

    uint32_t *devMarks = NULL;
    err = cudaMalloc((void **) &devMarks, prefixSize);
    handleError(err, __LINE__);

    uint64_t *devQueryIndices = NULL;
    err = cudaMalloc((void **) &devQueryIndices, qiSize);
    handleError(err, __LINE__);

    float3idx *devIndexed = NULL;
    err = cudaMalloc((void **) &devIndexed, indexedSize);
    handleError(err, __LINE__);

    float3idx *devData = NULL;
    err = cudaMalloc((void **) &devData, dataSize);
    handleError(err, __LINE__);

    uint64_t *devNearest = NULL;
    err = cudaMalloc((void **) &devNearest, nearestSize);
    handleError(err, __LINE__);

    float3 *devQueryNormals = NULL;
    err = cudaMalloc((void **) &devQueryNormals, queryNormalSize);
    handleError(err, __LINE__);

    float3 *devBucketNormals = NULL;
    err = cudaMalloc((void **) &devBucketNormals, bucketNormalSize);
    handleError(err, __LINE__);

    err = cudaMemcpy(devValues, values, valueSize, cudaMemcpyHostToDevice);
    handleError(err, __LINE__);

    err = cudaMemcpy(devQueryNormals, querynormals, queryNormalSize, cudaMemcpyHostToDevice);
    handleError(err, __LINE__);

    err = cudaMemcpy(devBucketNormals, bucketnormals, bucketNormalSize, cudaMemcpyHostToDevice);
    handleError(err, __LINE__);

    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);
    CoordinateSystem unit = unitSystem();

    float bucketScale = fmin(1.0, 2.0/compressionRate);
    float3 bucketNormalScaling = {bucketScale, bucketScale, bucketScale};
    float3 normalScaling = {1.0/compressionRate, 1.0/compressionRate, 1.0/compressionRate};
    float3 tangentScaling = {1.0, 1.0, 1.0};

    for (int bucket = 0; bucket < buckets; bucket++) {
        storeOriginalIndices(numElements, devValues, devIndexed);

        markInBucket(numData, numQueries, devQueryNormals, devBucketNormals, buckets, bucket, devMarks);

        thrust::sort_by_key(thrust::device, devMarks, devMarks + numElements, devIndexed);

        notInBucket(devMarks, numElements);
        int numOutside = thrust::reduce(thrust::device, devMarks, devMarks + numElements);
        
        int numQueriesInside = numQueries - numOutside;
        if(numQueriesInside == 0) {
            continue;
        }
        int numElementsInside = numData + numQueriesInside;
        float3 bucketnormal = normalize(bucketnormals[bucket]);
        float3 tangent0 = normalize(findArbitraryTangent(bucketnormal));
        float3 tangent1 = cross(bucketnormal, tangent0);
        CoordinateSystem bucketSpace = {
            mult(tangent0, tangentScaling),
            mult(bucketnormal, bucketNormalScaling),
            mult(tangent1, tangentScaling)
        };

        moveToCoordSpace(unit, bucketSpace, devIndexed, numElements, devIndexed);
        thrust::device_ptr<float3idx> dev_ptr = thrust::device_pointer_cast(devIndexed);
        float3pairidx xpair = thrust::minmax_element(dev_ptr, dev_ptr + numElementsInside, compare_x_idx());
        float3pairidx ypair = thrust::minmax_element(dev_ptr, dev_ptr + numElementsInside, compare_y_idx());
        float3pairidx zpair = thrust::minmax_element(dev_ptr, dev_ptr + numElementsInside, compare_z_idx());

        float3idx xmin = *(xpair.first);
        float3idx xmax = *(xpair.second);
        float minx = xmin.value.x;
        float maxx = xmax.value.x;

        float3idx ymin = *(ypair.first);
        float3idx ymax = *(ypair.second);
        float miny = ymin.value.y;
        float maxy = ymax.value.y;

        float3idx zmin = *(zpair.first);
        float3idx zmax = *(zpair.second);
        float minz = zmin.value.z;
        float maxz = zmax.value.z;

        float3 mins = {minx, miny, minz};
        float maxlen = fmax(maxx - minx, fmax(maxy - miny, maxz - minz));
        scaleValues(devIndexed, devIndexed, numElements, mins, maxlen, false);
        uint32_t *devReverseIndices = devMarks;
        reverseIndices(devIndexed, numElements, devReverseIndices);
        for (int j = 0; j < 5; ++j) {
            float shift = j*0.05;
            uint32_t intShift = (uint32_t) (shift * (1 << 21));
            getMortons(devIndexed, intShift, devMortons, numData, numQueriesInside);

            thrust::sort_by_key(thrust::device, devMortons, devMortons + numElementsInside, devIndexed);

            createPrefixList(devIndexed, devPrefixQueryIndex, numData, numElementsInside);

            pointCompaction(devIndexed, devMortons, devPrefixQueryIndex, devReverseIndices, devData, devQueryIndices, numData, numElementsInside);

            if(j == 0) {
                findCandidatesEllipsoid(
                    devQueryIndices,
                    devIndexed,
                    devData,
                    devNearest,
                    k,
                    numQueriesInside,
                    numData,
                    normalScaling,
                    tangentScaling,
                    bucketSpace,
                    devQueryNormals,
                    lambda
                );
            } else {
                mergeStepEllipsoid(
                    devNearest,
                    devIndexed,
                    devData,
                    devQueryIndices,
                    k,
                    numQueriesInside,
                    numData,
                    normalScaling,
                    tangentScaling,
                    bucketSpace,
                    devQueryNormals,
                    intShift,
                    lambda
                );
            }
        }
        scaleValues(devIndexed, devIndexed, numElements, mins, maxlen, true);
        moveToCoordSpace(bucketSpace, unit, devIndexed, numElements, devIndexed);
    }

    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    int64_t seconds = (int64_t) tval_result.tv_sec;
    int64_t micros = (int64_t) tval_result.tv_usec;
    uint64_t ms = seconds * 1000 + (micros / 1000);
    if(ms == 0) {
        ms = 1;    
    }
    uint64_t qsperms = numQueries / ms;
    printf("%d,%d,%u,%lu\n", numData, numQueries, k, qsperms);

    err = cudaMemcpy(nearest, devNearest, nearestSize, cudaMemcpyDeviceToHost);
    handleError(err, __LINE__);
    
   // for(int i = 0; i < numQueries; i++){
   //     float3 query = values[i+numData];
   //     fprintf(stderr, "%u ", i);
   //     for(int j = 0; j< k; j++){
   //         uint32_t valueIndex = (uint32_t) nearest[i*k+j];
   //         //if(valueIndex > numData) {
   //             fprintf(stderr, "%u ", (uint32_t) nearest[k*i+j]);
   //         //}
   //         //float3 value = values[valueIndex];
   //         //fprintf(stderr,"%u(%u) (%f,%f,%f) - ", (uint32_t) nearest[i*k+j], (uint32_t) (nearest[i*k+j] >> 32), value.x, value.y, value.z);
   //     }   
   //     fprintf(stderr,"\n");
   // }

    fprintf(stderr, "Time elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

    // Free device memory
    err = cudaFree(devValues);
    handleError(err, __LINE__);

    err = cudaFree(devMortons);
    handleError(err, __LINE__);

    err = cudaFree(devPrefixQueryIndex);
    handleError(err, __LINE__);

    err = cudaFree(devQueryIndices);
    handleError(err, __LINE__);

    err = cudaFree(devData);
    handleError(err, __LINE__);

    err = cudaFree(devIndexed);
    handleError(err, __LINE__);

    err = cudaFree(devNearest);
    handleError(err, __LINE__);

    err = cudaFree(devMarks);
    handleError(err, __LINE__);

    // Free host memory

    err = cudaDeviceReset();
    handleError(err, __LINE__);

    return EXIT_SUCCESS;

}

int nearestNeighbors(int numData, int numQueries, uint32_t k, float3 *values, uint64_t *nearest, const uint32_t lambda) {
    int numElements = numData + numQueries;

    size_t dataSize = numData * sizeof(uint4);
    uint4 *data = (uint4 *) malloc(dataSize);

    size_t qiSize = numQueries * sizeof(uint64_t);
    uint64_t *queryIndices = (uint64_t *) malloc(qiSize);

    size_t nearestSize = numQueries * k * sizeof(uint64_t);

    size_t mortonSize = numElements * sizeof(uint64_t);
    uint64_t *mortons = (uint64_t *) malloc(mortonSize);

    size_t prefixSize = numElements * sizeof(uint32_t);

    size_t intSize = numElements * sizeof(uint4);

    cudaError_t err = cudaSuccess;

    size_t valueSize = numElements * sizeof(float3);
    float3 *devValues = NULL;
    err = cudaMalloc((void **) &devValues, valueSize);
    handleError(err, __LINE__);

    float3 *uniqueValues = NULL;
    err = cudaMalloc((void **) &uniqueValues, valueSize);
    handleError(err, __LINE__);

    uint4 *devIntValues = NULL;
    err = cudaMalloc((void **) &devIntValues, intSize);
    handleError(err, __LINE__);

    uint64_t *devMortons = NULL;
    err = cudaMalloc((void **) &devMortons, mortonSize);
    handleError(err, __LINE__);

    uint64_t *uniqueMortons = NULL;
    err = cudaMalloc((void **) &uniqueMortons, mortonSize);
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

   // for(int i = 0; i < numElements; i++) {
   //     float3 v = values[i];
   //     fprintf(stderr, "%d: %f %f %f\n", i, v.x, v.y, v.z);
   // }

    err = cudaMemcpy(devValues, values, valueSize, cudaMemcpyHostToDevice);
    handleError(err, __LINE__);

    //float3 *result_end_f = thrust::unique_copy(thrust::device, devValues, devValues + numElements, uniqueValues);
    //fprintf(stderr, "Value duplicates %lu\n", numElements - (result_end_f - uniqueValues));

    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);


    thrust::device_ptr<float3> dev_ptr = thrust::device_pointer_cast(devValues);
    float3pair xpair = thrust::minmax_element(dev_ptr, dev_ptr + numElements, compare_x());
    float3pair ypair = thrust::minmax_element(dev_ptr, dev_ptr + numElements, compare_y());
    float3pair zpair = thrust::minmax_element(dev_ptr, dev_ptr + numElements, compare_z());

    float3 xmin = *(xpair.first);
    float3 xmax = *(xpair.second);
    float minx = xmin.x;
    float maxx = xmax.x;

    float3 ymin = *(ypair.first);
    float3 ymax = *(ypair.second);
    float miny = ymin.y;
    float maxy = ymax.y;

    float3 zmin = *(zpair.first);
    float3 zmax = *(zpair.second);
    float minz = zmin.z;
    float maxz = zmax.z;

    float3 mins = {minx, miny, minz};
    float maxlen = fmax(maxx - minx, fmax(maxy - miny, maxz - minz));
    printf("%f\n", maxlen);
    for(int j = 0; j < 5; ++j) {
        fprintf(stderr, "Iteration: %d\n",j);
        float shift = j*0.05;
        uint32_t intShift = (uint32_t) (shift * (1 << 21));
        scaleValues(devValues, devIntValues, intShift, numElements, mins, maxlen, false);
        getMortonsOld(devIntValues, devMortons, numData, numQueries);
        uint64_t *result_end = thrust::unique_copy(thrust::device, devMortons, devMortons + numElements, uniqueMortons);
        fprintf(stderr, "Num duplicates %lu\n", numElements - (result_end - uniqueMortons));

        //sort values in morton code order
        thrust::sort_by_key(thrust::device, devMortons, devMortons + numElements, devIntValues);
        createPrefixList(devIntValues, devPrefixQueryIndex, numData, numElements);
        pointCompactionOld(devIntValues, devMortons, devPrefixQueryIndex, devData, devQueryIndices, numData, numElements);

        if(j == 0) {
            findCandidates(devQueryIndices, devIntValues, devValues, devData, devNearest, k, numQueries, numData, lambda);
        } else {
            mergeStep(devNearest, devIntValues, devValues, devData, devQueryIndices, k, numQueries, numData, lambda);
        }
        scaleValues(devValues, devIntValues, intShift, numElements, mins, maxlen, true);
    }

    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    int64_t seconds = (int64_t) tval_result.tv_sec;
    int64_t micros = (int64_t) tval_result.tv_usec;
    uint64_t ms = seconds * 1000 + (micros / 1000);
    uint64_t qsperms = numQueries / ms;
    printf("%d,%d,%u,%lu\n", numData, numQueries, k, qsperms);

    err = cudaMemcpy(nearest, devNearest, nearestSize, cudaMemcpyDeviceToHost);
    handleError(err, __LINE__);

    //for(int i = 0; i < numQueries; i++){
    //    for(int j = 0; j< k; j++){
    //        fprintf(stderr,"%u(%u) ", (uint32_t) nearest[i*k+j], (uint32_t) (nearest[i*k+j] >> 32));
    //    }   
    //    fprintf(stderr,"\n");
    //}
    fprintf(stderr, "Time elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

    // Free device memory
    err = cudaFree(devValues);
    handleError(err, __LINE__);

    err = cudaFree(devIntValues);
    handleError(err, __LINE__);

    err = cudaFree(devMortons);
    handleError(err, __LINE__);

    err = cudaFree(devPrefixQueryIndex);
    handleError(err, __LINE__);

    err = cudaFree(devQueryIndices);
    handleError(err, __LINE__);

    err = cudaFree(devData);
    handleError(err, __LINE__);

    err = cudaFree(devNearest);
    handleError(err, __LINE__);

    // Free host memory
    free(data);
    free(queryIndices);
    free(mortons);

    err = cudaDeviceReset();
    handleError(err, __LINE__);

    return EXIT_SUCCESS;
}

/*
void calculateBounds(float3 *values, int numData, int numQuery, int dataElems, int end, float &minx, float &miny, float &minz, float &maxlen) {
    minx = FLT_MAX;
    miny = FLT_MAX;
    minz = FLT_MAX;

    float maxx = FLT_MIN;
    float maxy = FLT_MIN;
    float maxz = FLT_MIN;
    for(int i = 0; i < numData; ++i) {
        if(i >= dataElems) {
            break;
        }
        if(values[i].x < minx) {
            minx = values[i].x;
        }
        if(values[i].x > maxx) {
            maxx = values[i].x;
        }
        if(values[i].y < miny) {
            miny = values[i].y;
        }
        if(values[i].y > maxy) {
            maxy = values[i].y;
        }
        if(values[i].z < minz) {
            minz = values[i].z;
        }
        if(values[i].z > maxz) {
            maxz = values[i].z;
        }
    }

    for(int j = 0; j < numQuery; ++j) {
        int i = j + dataElems;
        if(i >= end) {
            break;
        }
        if(values[i].x < minx) {
            minx = values[i].x;
        }
        if(values[i].x > maxx) {
            maxx = values[i].x;
        }
        if(values[i].y < miny) {
            miny = values[i].y;
        }
        if(values[i].y > maxy) {
            maxy = values[i].y;
        }
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
*/

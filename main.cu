#include "knn.cuh"
#include "float3math.cuh"

int readCSV(const char *filename, float3 *values, int start, int end) {
    int numElements;
    FILE *file = fopen(filename, "r");
    if (file != NULL) {
        register float x,y,z,scale;
        fscanf(file, "%d %f\n", &numElements, &scale);
        for(int i = 0; i < numElements; ++i) {
            fscanf(file, "%f,%f,%f\n", &x, &y, &z);
            for(int j = i+start; j < end; j += numElements) {
                values[j].x = x*scale;
                values[j].y = y*scale;
                values[j].z = z*scale;
            }
        }
    }
    fclose(file);
    return numElements;
} 

float frand() {
    return rand()/(float)RAND_MAX;
}

float sfrand() {
    return 2.0*frand()-1.0;
}

void initValues(float3 *values, float3 *querynormals, int numElements, int numQueries) {
    for (int i = 0; i < numQueries; ++i) {
        querynormals[i].x = sfrand();
        querynormals[i].y = sfrand();
        querynormals[i].z = sfrand();
        querynormals[i] = normalize(querynormals[i]);
    }
    for (int i = 0; i < numElements; ++i) {
        values[i].x = sfrand();
        values[i].y = sfrand();
        values[i].z = sfrand();
    }
}

int main(int argc, char **argv) {
    int querySize = 1 << 17;
    int dataSize = 1 << 6;
    int kSize = 5;
    int size = dataSize + querySize;
    srand(time(0));
    size_t valueSize = size * sizeof(float3);
    float3 *values = (float3 *) malloc(valueSize);
//  float3 values[] = {
//      {1,1,1},{2,2,2},{1,3,1},{0,1,2},{5,5,5},{3,3,3},
//      {6,6,6}, {0,0,0}, {2,3,2}, {1,2,2}   
//  };
    size_t normalSize = querySize * sizeof(float3);
    float3 *querynormals = (float3 *) malloc(normalSize);
//    float3 querynormals[] = {
//        {0,1,1},{0,1,0},{0,1,0},{0,1,0}
//    };
    size_t nearestSize = querySize * kSize * sizeof(uint64_t);
    uint64_t *nearest = (uint64_t *) malloc(nearestSize);
    initValues(values, querynormals, size, querySize);
    const uint32_t lambda = 4;
    return nearestNeighborsEllipsoid(dataSize, querySize, kSize, values, querynormals, nearest, lambda, 4.0);
    //return nearestNeighbors(dataSize, querySize, kSize, values,nearest);
}

/*
// DK
int main(int argc, char **argv) {
    int numData = 20;
    int numQueries = 20;
    uint32_t k = 10;

    printf("Data,Queries,K,Queries per ms\n");
   //srand(time(NULL));
    uint64_t *nearest = NULL;
   for(int i = 0; k + i * 10 <= 160; ++i) {
       int times = (i == 0) ? 1 : 1;
       for(int j = 0; j < times; ++j) {
           int querySize = 1 << numQueries;
           int dataSize = 1 << numData;
           int kSize = k + i * 10;
           int size = dataSize + querySize;
           size_t valueSize = size * sizeof(float3);
           size_t normalSize = querySize * sizeof(float3);
           float3 *values = (float3 *) malloc(valueSize);
           float3 *querynormals = (float3 *) malloc(normalSize);
           if(argc == 3) {
               int dataPoints = readCSV(argv[1], values, 0, dataSize);
               int queryPoints = readCSV(argv[2], values, dataSize, size);
               //calculateBounds(values, dataPoints, queryPoints, dataSize, size, minx, miny, minz, maxlen);
           } else {
               initValues(values, querynormals, size, querySize);
           }
	       size_t nearestSize = querySize * kSize * sizeof(uint64_t);
	       nearest = (uint64_t *) realloc(nearest, nearestSize);
           //nearestNeighbors(dataSize, querySize, kSize, values, nearest, minx, miny, minz, maxlen);
           nearestNeighborsEllipsoid(dataSize, querySize, kSize, values, querynormals, nearest);
       }
   }
}
*/

/*
// dq
int main(int argc, char **argv) {
    int numData = 21;
    int numQueries = 7;
    uint32_t k = 64;

    printf("Data,Queries,K,Queries per ms\n");
   //srand(time(NULL));
    uint64_t *nearest = NULL;
   for(int i = 0; numQueries + i <= 21; ++i) {
       int times = (i == 0) ? 1 : 1;
       for(int j = 0; j < times; ++j) {
           int querySize = 1 << (numQueries + i);
           int dataSize = 1 << numData;
           int kSize = k;
           int size = dataSize + querySize;
           size_t valueSize = size * sizeof(float3);
           size_t normalSize = querySize * sizeof(float3);
           float3 *values = (float3 *) malloc(valueSize);
           float3 *querynormals = (float3 *) malloc(normalSize);
           if(argc == 3) {
               int dataPoints = readCSV(argv[1], values, 0, dataSize);
               int queryPoints = readCSV(argv[2], values, dataSize, size);
               //calculateBounds(values, dataPoints, queryPoints, dataSize, size, minx, miny, minz, maxlen);
           } else {
               initValues(values, querynormals, size, querySize);
           }
	       size_t nearestSize = querySize * kSize * sizeof(uint64_t);
	       nearest = (uint64_t *) realloc(nearest, nearestSize);
           //nearestNeighbors(dataSize, querySize, kSize, values, nearest, minx, miny, minz, maxlen);
           nearestNeighborsEllipsoid(dataSize, querySize, kSize, values, querynormals, nearest);
       }
   }
}
*/
/*
// dd
int main(int argc, char **argv) {
    int numData = 7;
    int numQueries = 20;
    uint32_t k = 64;

    printf("Data,Queries,K,Queries per ms\n");
   //srand(time(NULL));
    uint64_t *nearest = NULL;
   for(int i = 0; numData + i <= 21; ++i) {
       int times = (i == 0) ? 1 : 1;
       for(int j = 0; j < times; ++j) {
           int querySize = 1 << numQueries;
           int dataSize = 1 << (numData + i);
           int kSize = k;
           int size = dataSize + querySize;
           size_t valueSize = size * sizeof(float3);
           size_t normalSize = querySize * sizeof(float3);
           float3 *values = (float3 *) malloc(valueSize);
           float3 *querynormals = (float3 *) malloc(normalSize);
           if(argc == 3) {
               int dataPoints = readCSV(argv[1], values, 0, dataSize);
               int queryPoints = readCSV(argv[2], values, dataSize, size);
               //calculateBounds(values, dataPoints, queryPoints, dataSize, size, minx, miny, minz, maxlen);
           } else {
               initValues(values, querynormals, size, querySize);
           }
	       size_t nearestSize = querySize * kSize * sizeof(uint64_t);
	       nearest = (uint64_t *) realloc(nearest, nearestSize);
           //nearestNeighbors(dataSize, querySize, kSize, values, nearest, minx, miny, minz, maxlen);
           nearestNeighborsEllipsoid(dataSize, querySize, kSize, values, querynormals, nearest);
       }
   }
}
*/

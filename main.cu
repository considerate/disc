#include "knn.cuh"

int main(int argc, char **argv) {
    int numData = 20;
    int numQueries = 20;
    uint32_t k = 10;


    printf("Data,Queries,K,Queries per ms\n");
   //srand(time(NULL));
   for(int i = 0; k + i * 10 <= 160; ++i) {
       int times = (i == 0) ? 4 : 1;
       for(int j = 0; j < times; ++j) {
           int querySize = 1 << numQueries;
           int dataSize = 1 << numData;
           int kSize = k + i * 10;
           int size = dataSize + querySize;
           size_t valueSize = size * sizeof(float3);
           float3 *values = (float3 *) malloc(valueSize);
           float minx, miny, minz, maxlen;
           if(argc == 3) {
               int dataPoints = readCSV(argv[1], values, 0, dataSize);
               int queryPoints = readCSV(argv[2], values, dataSize, size);
               calculateBounds(values, dataPoints, queryPoints, dataSize, size, minx, miny, minz, maxlen);
           } else {
               initValues(values, minx, miny, minz, maxlen, size);
           }
	    size_t nearestSize = numQueries * k * sizeof(uint64_t);
	    uint64_t *nearest = (uint64_t *) malloc(nearestSize);
           nearestNeighbors(dataSize, querySize, kSize, values, nearest, minx, miny, minz, maxlen);
       }
   }
}

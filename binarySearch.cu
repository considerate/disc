#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>
#include <stdint.h>
#include <assert.h>

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

int main() {
    uint64_t data[] = {1,2,4,6,7,9};

    for(int i=0; i<6; i++)
    {
        uint32_t result = binarySearch(data, data[i], 6);
        //printf("%u\n",result);
    }

    printf("%u\n",binarySearch(data, 0, 6));
    printf("%u\n",binarySearch(data, 1, 6));
    printf("%u\n",binarySearch(data, 2, 6));
    printf("%u\n",binarySearch(data, 3, 6));
    printf("%u\n",binarySearch(data, 4, 6));
    printf("%u\n",binarySearch(data, 9, 6));
    printf("%u\n",binarySearch(data, 10, 6));

    return 0;
}

template<class T>
__device__ T plus_scan(T *x)
{
    unsigned int i = threadIdx.x; // id of thread executing this instance
    unsigned int n = blockDim.x;  // total number of threads in this block
    unsigned int offset;          // distance between elements to be added

    for( offset = 1; offset < n; offset *= 2) {
        T t;

        if ( i >= offset ) {
            t = x[i-offset];
        }
        
        __syncthreads();

        if ( i >= offset ) {
            x[i] = t + x[i];      // i.e., x[i] = x[i] + x[i-1]
        }

        __syncthreads();
    }
    return x[i];
}

__device__ void partition_by_bit(unsigned int *keys, uint64_t *values, unsigned int bit)
{
    unsigned int i = threadIdx.x;
    unsigned int size = blockDim.x;
    uint64_t x_i = values[i];          // value of integer at position i
    unsigned int k_i = keys[i];
    uint64_t p_i = (x_i >> bit) & 1;   // value of bit at position bit

    values[i] = p_i;  

    __syncthreads();

    uint64_t T_before = plus_scan(values);
    uint64_t T_total  = values[size-1];
    uint64_t F_total  = size - T_total;

    __syncthreads();

    if ( p_i ) {
        values[T_before-1 + F_total] = x_i;
        //keys[T_before-1 + F_total] = k_i;
    }
    else {
        values[i - T_before] = x_i;
        //keys[i - T_before] = k_i;
    }
}

__device__ void radix_sort(unsigned int *keys, uint64_t *values)
{
    int  bit;
    for( bit = 0; bit < 64; ++bit )
    {
        partition_by_bit(keys, values, bit);
        __syncthreads();
    }
}

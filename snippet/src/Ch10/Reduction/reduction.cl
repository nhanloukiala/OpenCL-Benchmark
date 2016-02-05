
#define BLOCK_SIZE 256
/*
// This reduction uses interleaved addressing
__kernel void reduce0(__global uint* input, __global uint* output, __local uint* sdata) {
    unsigned int tid = get_local_id(0);
    unsigned int bid = get_group_id(0);
    unsigned int gid = get_global_id(0);
    unsigned int blockSize = get_local_size(0);

    sdata[tid] = input[gid];

    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int s = 1; s < BLOCK_SIZE; s <<= 1) {
        // This has a slight problem, the %-operator is rather slow
        // and causes divergence within the wavefront as not all threads
        // within the wavefront is executing. 
        if(tid % (2*s) == 0) 
        {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem
    if(tid == 0) output[bid] = sdata[0];
}

__kernel void reduce1(__global uint* input, __global uint* output, __local uint* sdata) {
    unsigned int tid = get_local_id(0);
    unsigned int bid = get_group_id(0);
    unsigned int gid = get_global_id(0);
    unsigned int blockSize = get_local_size(0);

    sdata[tid] = input[gid];

    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int s = 1; s < BLOCK_SIZE; s <<= 1) {
        // Actually, this is pretty good but we have a new problem
        // and they are bank conflicts in the shared memory 
        int index = 2 * s * tid;
        if(index < BLOCK_SIZE) 
        {
            sdata[index] += sdata[index + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem
    if(tid == 0) output[bid] = sdata[0];
}

__kernel void reduce2(__global uint* input, __global uint* output, __local uint* sdata) {
    unsigned int tid = get_local_id(0);
    unsigned int bid = get_group_id(0);
    unsigned int gid = get_global_id(0);
    unsigned int blockSize = get_local_size(0);

    sdata[tid] = input[gid];

    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int s = BLOCK_SIZE/2; s > 0 ; s >>= 1) {
        // Notice that half of threads are already idle on first iteration
        // and with each iteration, its halved again. Work efficiency isn't very good
        // now
        if(tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem
    if(tid == 0) output[bid] = sdata[0];
}

__kernel void reduce3(__global uint* input, __global uint* output, __local uint* sdata) {
    unsigned int tid = get_local_id(0);
    unsigned int bid = get_group_id(0);
    unsigned int gid = get_global_id(0);

    // To mitigate the problem of idling threads in 'reduce2' kernel,
    // we can halve the number of blocks while each work-item loads
    // two elements instead of one into shared memory
    unsigned int index = bid*(BLOCK_SIZE*2) + tid;
    sdata[tid] = input[index] + input[index+BLOCK_SIZE];

    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int s = BLOCK_SIZE/2; s > 0 ; s >>= 1) {
        // Notice that half of threads are already idle on first iteration
        // and with each iteration, its halved again. Work efficiency isn't very good
        // now
        if(tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem
    if(tid == 0) output[bid] = sdata[0];
}
*/
__kernel void reduce4(__global uint* input, __global uint* output, __local uint* sdata) {
    unsigned int tid = get_local_id(0);
    unsigned int bid = get_group_id(0);
    unsigned int gid = get_global_id(0);
    unsigned int blockSize = get_local_size(0);

    unsigned int index = bid*(BLOCK_SIZE*2) + tid;
    sdata[tid] = input[index] + input[index+BLOCK_SIZE];

    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int s = BLOCK_SIZE/2; s > 64 ; s >>= 1) {
        // Unrolling the last wavefront and we cut 7 iterations of this
        // for-loop while we practice wavefront-programming
        if(tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid < 64) {
        if (blockSize >= 128) sdata[tid] += sdata[tid + 64];
        if (blockSize >=  64) sdata[tid] += sdata[tid + 32];
        if (blockSize >=  32) sdata[tid] += sdata[tid + 16];
        if (blockSize >=  16) sdata[tid] += sdata[tid +  8];
        if (blockSize >=   8) sdata[tid] += sdata[tid +  4];
        if (blockSize >=   4) sdata[tid] += sdata[tid +  2];
        if (blockSize >=   2) sdata[tid] += sdata[tid +  1];
    }
    // write result for this block to global mem
    if(tid == 0) output[bid] = sdata[0];
}


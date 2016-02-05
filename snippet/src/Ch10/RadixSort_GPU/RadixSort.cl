#define bitsbyte 8
#define R (1 << bitsbyte)

__kernel void computeHistogram(__global const uint* data,
                               __global uint* buckets,
                               uint shiftBy,
                               __local uint* sharedArray) {

    size_t localId = get_local_id(0);
    size_t globalId = get_global_id(0);
    size_t groupId = get_group_id(0);
    size_t groupSize = get_local_size(0);
    
    /* Initialize shared array to zero i.e. sharedArray[0..63] = {0}*/
    sharedArray[localId] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    /* Calculate thread-histograms local/shared memory range from 32KB to 64KB */

    uint result= (data[globalId] >> shiftBy) & 0xFFU;
    atomic_inc(sharedArray+result);
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    /* Copy calculated histogram bin to global memory */
    
    uint bucketPos = groupId  * groupSize + localId ;
    buckets[bucketPos] = sharedArray[localId];
}

__kernel void rankNPermute(__global const uint* unsortedData,
                           __global const uint* scannedHistogram,
                           uint shiftCount,
                           __local ushort* sharedBuckets,
                           __global uint* sortedData) { 

    size_t groupId = get_group_id(0);
    size_t idx = get_local_id(0);
    size_t gidx = get_global_id(0);
    size_t groupSize = get_local_size(0);
    
    /* There are now GROUP_SIZE * RADIX buckets and we fill
       the shared memory with those prefix-sums computed previously
     */ 
    for(int i = 0; i < R; ++i)
    {
        uint bucketPos = groupId * R * groupSize + idx * R + i;
        sharedBuckets[idx * R + i] = scannedHistogram[bucketPos];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
   
    /* Using the idea behind COUNTING-SORT to place the data values in its sorted
       order based on the current examined key
     */
    for(int i = 0; i < R; ++i)
    {
        uint value = unsortedData[gidx * R + i];
        value = (value >> shiftCount) & 0xFFU;
        uint index = sharedBuckets[idx * R + value];
        sortedData[index] = unsortedData[gidx * R + i];
        sharedBuckets[idx * R + value] = index + 1;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


__kernel void blockScan(__global uint *output,
                        __global uint *histogram,
                        __local uint* sharedMem,
                        const uint block_size,
                        __global uint* sumBuffer) {
      int idx = get_local_id(0);
	  int gidx = get_global_id(0);
	  int gidy = get_global_id(1);
	  int bidx = get_group_id(0);
	  int bidy = get_group_id(1);
	  
	  int gpos = (gidx << bitsbyte) + gidy;
	  int groupIndex = bidy * (get_global_size(0)/block_size) + bidx;
	  
	  /* Cache the histogram buckets into shared memory 
         and memory reads into shared memory is coalesced
      */
	  sharedMem[idx] = histogram[gpos];
	  barrier(CLK_LOCAL_MEM_FENCE);

    /* 
       Build the partial sums sweeping up the tree using 
       the idea of Hillis and Steele in 1986
     */
	uint cache = sharedMem[0];
	for(int stride = 1; stride < block_size; stride <<= 1)
	{
		if(idx>=stride)
		{
			cache = sharedMem[idx-stride]+sharedMem[idx];
		}
		barrier(CLK_LOCAL_MEM_FENCE); // all threads are blocked here

		sharedMem[idx] = cache;
		barrier(CLK_LOCAL_MEM_FENCE);
	}

    /* write the array of computed prefix-sums back to global memory */
	if(idx == 0)
	{	
        /* store the value in sum buffer before making it to 0 */ 	
	    sumBuffer[groupIndex] = sharedMem[block_size-1];
		output[gpos] = 0;
	}
	else
	{
		output[gpos] = sharedMem[idx-1];
	}
}   

__kernel void unifiedBlockScan(__global uint *output,
                               __global uint *input,
                               __local uint* sharedMem,
                               const uint block_size) {

    int id = get_local_id(0);
 	int gid = get_global_id(0);
	int bid = get_group_id(0);
	
    /* Cache the computational window in shared memory */
	sharedMem[id] = input[gid];

	uint cache = sharedMem[0];

    /* build the sum in place up the tree */
	for(int stride = 1; stride < block_size; stride <<= 1)
	{
		if(id>=stride)
		{
			cache = sharedMem[id-stride]+sharedMem[id];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		sharedMem[id] = cache;
		barrier(CLK_LOCAL_MEM_FENCE);
		
	}
    /*write the results back to global memory */
	if(id == 0) {	
		output[gid] = 0;
	} else {
		output[gid] = sharedMem[id-1];
	}
} 
  
__kernel void blockPrefixSum(__global uint* output,
                             __global uint* input,
                             __global uint* summary,
                             int stride) {

     int gidx = get_global_id(0);
     int gidy = get_global_id(1);
     int Index = gidy * stride +gidx;
     output[Index] = 0;
    
      // Notice that you don't need memory fences in this kernel
      // because there is no race conditions and the assumption
      // here is that the hardware schedules the blocks with lower 
      // indices first before blocks with higher indices
     if(gidx > 0)
     {
         for(int i =0;i<gidx;i++)
             output[Index] += input[gidy * stride +i];
     }
     // Write out all the prefix sums computed by this block 
     if(gidx == (stride - 1)) 
         summary[gidy] = output[Index] + input[gidy * stride + (stride -1)];
} 
  
__kernel void blockAdd(__global uint* input,
                       __global uint* output,
                       uint stride) {

	  int gidx = get_global_id(0);
	  int gidy = get_global_id(1);
	  int bidx = get_group_id(0);
	  int bidy = get_group_id(1);
	
	  
	  int gpos = gidy + (gidx << bitsbyte);
	 
	  int groupIndex = bidy * stride + bidx;
	  
	  uint temp;
	  temp = input[groupIndex];
	  
	  output[gpos] += temp;
}
   
__kernel void mergePrefixSums(__global uint* input,
                        __global uint* output) {
 
   int gidx = get_global_id(0);
   int gidy = get_global_id(1);
   int gpos = gidy + (gidx << bitsbyte );
   output[gpos] += input[gidy];
}           

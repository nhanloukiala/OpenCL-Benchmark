
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#define BIN_SIZE 256

/**
 * @param   data - input data pointer
 * @param   sharedArray - shared array for thread-histogram bins
 * @param   binResult - block-histogram array
 */

__kernel
void histogram256(__global const uint4* data,
                  __local uchar* sharedArray,
                  __global uint* binResult)
{
    size_t localId = get_local_id(0);
    size_t globalId = get_global_id(0);
    size_t groupId = get_group_id(0);
    size_t groupSize = get_local_size(0);
 
//     This is a form of the optimized version of the same program that 
//     makes use of the memory banks in the device (works only for GPUs)
//     and was meant to help reduce bank conflicts because of the presence
//     of 'sharedArray' which we use to read from/write to.
// 
    int offSet1 = localId & 31;
    int offSet2 = 4 * offSet1;      //which element to access in one bank.
    int bankNumber = localId >> 5;     //bank number
    
//    initialize shared array to zero via assignment of (int)(0) to uchar4(0)
	__local uchar4 * input = (__local uchar4*)sharedArray;
    
//    
//     memset's the local array of 32-kb to 0
//     groupSize = 128 i.e. [0..127]
//     i = 0, input[0..127]   = 0
//     i = 1, input[128..255] = 0
//     i = 2, input[256..383] = 0
//     ...
//     i = 63, input[8064..8191] = 0
//     but since input is uchar4 hence its 32-KB
//     
    for(int i = 0; i < 64; ++i)
        input[groupSize * i + localId] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);


//    calculate thread-histograms
//	uint4 value = data[groupId * groupSize *64  + localId];
    for(int i = 0; i < 64; i++)
    {
       uint4 value =  data[groupId * groupSize * BIN_SIZE/4 + i * groupSize + localId];
        
 
//        value is a uint4 - vector of 4 'unsigned int'
//        and this loop walks through the entire 32-KB of locally shared
//        data and populates the histogram.
//        
//        Note: one possible issue is the fact that bank conflicts can occur
//              when any of computations below reference the same memory bank.
 
       sharedArray[value.s0 * 128 + offSet2 + bankNumber]++;
       sharedArray[value.s1 * 128 + offSet2 + bankNumber]++;
       sharedArray[value.s2 * 128 + offSet2 + bankNumber]++;
       sharedArray[value.s3 * 128 + offSet2 + bankNumber]++;
    }
    barrier(CLK_LOCAL_MEM_FENCE); 
    
//    merge all thread-histograms into block-histogram

    if(localId == 0) {
        for(int i = 0; i < BIN_SIZE; ++i) {
            uint result = 0;
            for(int j = 0; j < 128; ++j)  {
                result += sharedArray[i * 128 + j];
            }
            binResult[groupId * BIN_SIZE + i] = result;
        }
    }
}

// only 1 thread executing a 256 block
__kernel
void histogram256_1threadPerBlock(__global const unsigned int* data,
                                  __local uint* sharedArray,
                                  __global uint* binResult)
{
    size_t localId = get_local_id(0);
    size_t globalId = get_global_id(0);
    size_t groupId = get_group_id(0);
    size_t groupSize = get_local_size(0);
    size_t numOfGroups = get_num_groups(0);
   
	__local uint * input = (__local uint*)sharedArray; // assume its 256 * sizeof(cl_uint) and range from [0..255]
    
    for(int i = 0; i < BIN_SIZE; ++i)
        input[i] = 0;
  
    barrier(CLK_LOCAL_MEM_FENCE);
 
    for(int i = 0; i < BIN_SIZE; ++i)
        input[ data[i] ]++;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId == 0) 
    for(int i = 0; i < BIN_SIZE; ++i) {
        uint t = input[i]; 
        atomic_add(binResult + i, t);
    }
} 
 

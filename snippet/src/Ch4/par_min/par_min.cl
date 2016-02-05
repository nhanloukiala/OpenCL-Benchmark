
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable 
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable 

__kernel void par_min(__global uint4* src,
                      __global uint * globalMin,
                      __local  uint * localMin,
                      int             numOfItems,
                      uint            device) {

    uint count = ( numOfItems / 4) / get_global_size(0);
    uint index = (dev == 0) ? get_global_id(0) * count: get_global_id(0);
    
    uint stride = (dev == 0) ? 1 : get_global_size(0);
    uint partialMin = (uint) -1;

    for(int i = 0; i < count; ++i, index += stride) {
        partialMin = min(partialMin, src[index].x);
        partialMin = min(partialMin, src[index].y);
        partialMin = min(partialMin, src[index].z);
        partialMin = min(partialMin, src[index].w);
    }

    if(get_local_id(0) == 0) localMin[0] = (uint) -1;

    barrier(CLK_LOCAL_MEM_FENCE);

    atomic_min(localMin, globalMin);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) == 0) globalMin[ get_group_id[0] ] = localMin[0];
}

__kernel void reduce(__global uint4* src,
                     __global uint * globalMin) {
    atom_min(globalMin, globalMin[get_global_id(0)]);
}

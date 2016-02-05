__kernel void permutate(__global float8* a,
                        __global float8* b,
                        __global uint16* mask,
                        __global float16* result) {
    //uint16 mask = (uint16)(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    //uint16 mask0 = (uint16)(0,1,2,3,0,1,2,3,0,1,2,3,12,13,14,15);

    uint id = get_global_id(0);
    float8 in1 = a[id];
    float8 in2 = b[id];
    result[id] = shuffle2(in1, in2, *mask);
}

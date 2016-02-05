__kernel void filter_by_selection(__global float8* a,
                                  __global float8* b,
                                  __global float8* result) {
    //int8 mask = (0,0,0,0,0,0,0,0);
    //int8 mask = (-1,-1,-1,-1,-1,-1,-1,-1);
    uint8 mask = (uint8)(0,-1,0,-1,0,-1,0,-1);  

    uint id = get_global_id(0);
    float8 in1 = a[id+1];
    float8 in2 = b[id+1];
    result[id+1] = select(in1, in2, mask);
}

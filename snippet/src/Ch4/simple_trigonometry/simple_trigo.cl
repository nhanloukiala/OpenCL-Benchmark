__kernel void find_unit_circles(__global float16* a,
                                __global float16* b,
                                __global float16* result) {
    uint id = get_global_id(0);
    float16 x = a[id];
    float16 y = b[id];
    float16 tresult = sin(x) * sin(x) + cos(y) * cos(y) ;
    result[id] = tresult ;
}

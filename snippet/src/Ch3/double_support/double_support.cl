
#ifdef fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

__kernel void add3(__global float* a, __global float* b, __global float* out) {
    int id = get_global_id(0);
#ifdef fp64
    double d = a[id] + b[id];
    out[id] = d;
#else
    out[id] = a[id] + b[id];
#endif

}

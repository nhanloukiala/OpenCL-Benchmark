// OpenCL 1.1 doesn't appear to allow 
// header files to be included in kernel files (*.cl)

__kernel void copyNPaste(__global float* in, __global float8* out) {
    size_t id = get_global_id(0);
    size_t index = id*sizeof(float8);
    float8 t = vload8(index, in);
    out[index].s0 = t.s0;
    out[index].s1 = t.s1;
    out[index].s2 = t.s2;
    out[index].s3 = t.s3;
    out[index].s4 = t.s4;
    out[index].s5 = t.s5;
    out[index].s6 = t.s6;
    out[index].s7 = t.s7;
}

// This version is the shorter form of the previously defined vector 
// kernel (which states the data transfers explicitly)
__kernel void copyNPaste_2(__global float8* in, __global float8* out) {
    size_t id = get_global_id(0);
    out[id] = in[id];
}

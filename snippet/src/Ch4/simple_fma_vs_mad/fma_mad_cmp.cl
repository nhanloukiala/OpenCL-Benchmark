__kernel void mad_test(__global float* a,
                       __global float* b,
                       __global float* c,
                       __global float* result) {

    size_t id = get_global_id(0);
    result[get_global_id(0)] = mad(a[id], b[id], c[id]);
}
__kernel void fma_test(__global float* a,
                       __global float* b,
                       __global float* c,
                       __global float* result) {

    size_t id = get_global_id(0);
    result[get_global_id(0)] = fma(a[id], b[id], c[id]);
}

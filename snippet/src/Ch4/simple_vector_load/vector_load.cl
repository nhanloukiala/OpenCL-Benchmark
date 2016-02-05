//
// This kernel loads 64-elements using a single thread/work-item
// into its __private memory space and writes it back out
__kernel void wideDataTransfer(__global float* in, __global float* out) {
    size_t id = get_group_id(0) * get_local_size(0) + get_local_id(0);
    size_t STRIDE = 16;
    size_t offsetA = id;

    prefetch(in + (get_local_id(0) * 16), 16);

    barrier(CLK_LOCAL_MEM_FENCE);

    float16 A = vload16(offsetA, in);
    float a[16]; 
    a[0] = A.s0;
    a[1] = A.s1;
    a[2] = A.s2;
    a[3] = A.s3;
    a[4] = A.s4;
    a[5] = A.s5;
    a[6] = A.s6;
    a[7] = A.s7;
    a[8] = A.s8;
    a[9] = A.s9;
    a[10] = A.sa;
    a[11] = A.sb;
    a[12] = A.sc;
    a[13] = A.sd;
    a[14] = A.se;
    a[15] = A.sf;

    for( int i = 0; i < 16; ++i ) {
        out[offsetA*STRIDE+i] = a[i];
    }
}

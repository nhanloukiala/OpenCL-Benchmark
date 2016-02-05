__kernel void sample(__global float* A, __global float* B, __global float* C)
{
    int base = 4*get_global_id(0);
 
    C[base+0] = A[base+0] + B[base+0];
    C[base+1] = A[base+1] - B[base+1];
    C[base+2] = A[base+2] * B[base+2];
    C[base+3] = A[base+3] / B[base+3];
}
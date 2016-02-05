__kernel void simpleAdd_2(__global float *a,
                   __global float *b,
                   __global float *c) {
   
   *c = *a + *b;
}

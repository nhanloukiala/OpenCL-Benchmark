__kernel void simpleAdd(__global float *a,
                   __global float *b,
                   __global float *c) {
   
   *c = *a + *b;
}

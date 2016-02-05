/* 
  This is a direct translation of the Matrix-matrix multiplication
  algorithm into OpenCL form.
*/

__kernel void mmmult(int widthB, 
                     int heightA, 
                      __global int* A, 
                      __global int* B, 
                      __global int* C,
                      __local  int* shared) {

    int i = get_global_id(0);
    int id = get_local_id(0);
    int size = get_local_size(0);
    int tmp = 0;

    int tmpData[1024];

    if (i < heightA) {
        for(int k = 0; k < widthB; ++k ) {
            tmpData[k] = A[i*heightA + k];
        }

        for(int j = 0; j < heightA; ++j) {
            for(int k = id; k < widthB; k+=size) 
                shared[k] = B[k*widthB +j];
            barrier(CLK_LOCAL_MEM_FENCE);

            tmp = 0;
            for(int k = 0; k < widthB; ++k) {
                tmp += tmpData[k] * shared[k];
            }
            C[i*heightA + j] = tmp;
        }
    }
}



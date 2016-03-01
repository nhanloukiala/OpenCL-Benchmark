#define WIDTH 1024
#define DATA_TYPE float4

/* 
  The following macros are convenience 'functions' 
  for striding across a 2-D array of coordinates (x,y)
  by a factor which happens to be the width of the block
  i.e. WIDTH
*/
#define A(x,y) A[(x)* WIDTH + (y)]
#define C(x,y) C[(x)* WIDTH + (y)]
__kernel void copy2Dfloat4(__global DATA_TYPE *A, __global DATA_TYPE *C)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int z = get_local_id(0);
    int t = get_group_id(0);
    int w = get_local_size(0);
    int m = get_global_offset(0);

    int inter = z + t * w + m;


    // its like a vector load/store of 4 elements
    C(x,y) = x == inter;
}


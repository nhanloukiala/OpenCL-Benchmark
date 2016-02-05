__kernel void MatVecMultUsingDotFn(__global float4* matrix,
                               __global float4* vector,
                               __global float* result) {
    size_t i = get_group_id(0) * get_local_size(0) + get_local_id(0);
    result[i] = dot(matrix[i], vector[0]);
}


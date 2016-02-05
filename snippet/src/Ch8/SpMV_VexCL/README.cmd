
CL_BUILD_PROGRAM_FAILURE error when using double precision floating point computations
Posted on 2012/10/19 by Guillaume Poirier   

Q:
When I execute my CAPS Compiler application, with the OpenCL backend, I get a runtime log like the following:

ERROR:       clBuildProgram() failed: CL_BUILD_PROGRAM_FAILURE (-11)
ERROR:       OCL Build log : 
 "/tmp/mtest_mMYTFP.halt.f90", line 14: error: identifier "double" is undefined

A:
You application uses double precision computations, which means that the OpenCL extension cl_khr_fp64 has to be used. It turns out that this extension is not supported by all AMD GPUs. This is why the OpenCL program does not compile.

You can check which extensions are supported by your card with the clinfo program, located in $AMDAPPSDKROOT/bin/x86_64/clinfo

Fortunately, most AMD cards supports the extension cl_amd_fp64, which is AMD-specific, and allows most of times to perform floating point, double precision computations on AMD cards, like cl_khr_fp64.
AMD’s OpencCL SDK allows to use the cl_amd_fp64 extension in lieu of cl_khr_fp64 if you define the environment variable CL_KHR_FP64 to 1. This will allow your application to run.
example:
export CL_KHR_FP64=1
Posted in OpenCL, Runtime 
===============================================================================================
tayboonl@ubuntu:~/PACKT_OpenCL_Book/src/Ch8/spmv$ ./SpMV 
#if defined(cl_khr_fp64)
#  pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif defined(cl_amd_fp64)
#  pragma OPENCL EXTENSION cl_amd_fp64: enable
#endif
typedef double real;
kernel void gather_vals_to_send(
    ulong n,
    global const real *vals,
    global const ulong *cols_to_send,
    global real *vals_to_send
    )
{
    size_t i = get_global_id(0);
    if (i < n) vals_to_send[i] = vals[cols_to_send[i]];
}

"/tmp/OCLLUdZc8.cl", line 6: error: identifier "double" is undefined
  typedef double real;
          ^

1 error detected in the compilation of "/tmp/OCLLUdZc8.cl".

Internal error: clc compiler invocation failed.

terminate called after throwing an instance of 'cl::Error'
  what():  clBuildProgram
Aborted (core dumped)

=========================================================================
You need to find a GPU device that supports double precision as
VexCL does not have support for double-precision.

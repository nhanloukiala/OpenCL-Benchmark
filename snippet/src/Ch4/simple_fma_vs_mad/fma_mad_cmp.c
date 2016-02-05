#ifdef APPLE
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include "fma_mad_cmp_config.h"

#define NDEVS   2
#define DATA_SIZE 128;

int valuesOK(cl_float* src, cl_float* dst, size_t length) {
#ifdef DEBUG 
  printf("Checking data of size: %d\n", length);
#endif
  for(int i =0; i < length; ++i) {
#ifdef DEBUG 
    printf("(%d) src=%f, dest=%f\n", i, src[i], dst[i]);
#endif
    if (src[i] != dst[i]) return 0;
  }
  return 1;
}

void loadProgramSource(const char** files,
                       size_t length,
                       char** buffer,
                       size_t* sizes) {
	   /* Read each source file (*.cl) and store the contents into a temporary datastore */
	   for(size_t i=0; i < length; i++) {
	      FILE* file = fopen(files[i], "r");
	      if(file == NULL) {
	         perror("Couldn't read the program file");
	         exit(1);   
	      }
	      fseek(file, 0, SEEK_END);
	      sizes[i] = ftell(file);
	      rewind(file); // reset the file pointer so that 'fread' reads from the front
	      buffer[i] = (char*)malloc(sizes[i]+1);
	      buffer[i][sizes[i]] = '\0';
	      fread(buffer[i], sizeof(char), sizes[i], file);
	      fclose(file);
	   }
}



/*
    This program should run on any device that supports OpenCL 1.1

    Note: In the event that it runs on the CPU, then we cannot assign
          a local_work_size > 1 since the CPU forbids it. In the GPU, 
          that's an entirely different story :) whew
*/

int main(int argc, char** argv) {
  cl_platform_id platform;
  int dev;
  cl_device_type devs[NDEVS] = { CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU };

  cl_float *src_ptr;
  unsigned int numOfItems = DATA_SIZE;

  src_ptr = (cl_float*) malloc( numOfItems * sizeof(cl_float));

  for( int i = 0; i < numOfItems; ++i) {
    src_ptr[i] = 4;
  }

  // Get the supported platforms

  clGetPlatformIDs(1, &platform, NULL);

  for( dev = 0; dev < NDEVS; dev++ ) { 
    cl_device_id     device;
    cl_context       context;
    cl_command_queue cQMAD;
    cl_command_queue cQFMA;
    cl_program       program;
    cl_kernel        mad;
    cl_kernel        fma;

    cl_mem           a_buffer;
    cl_mem           b_buffer;
    cl_mem           c_buffer;
    cl_mem           fma_res_buffer;
    cl_mem           mad_res_buffer;
    
    cl_float          *fma_dst_ptr;
    cl_float          *mad_dst_ptr;
    
    clGetDeviceIDs(platform, devs[dev], 1, &device, NULL);
    cl_float compute_units;
    size_t  global_work_size;
    size_t  local_work_size;
    size_t  num_groups;

    clGetDeviceInfo( device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_float), &compute_units, NULL);

    // Since the CPU doesn't permit parallel executing threads per core
    // hence we assign 1 thread to per core
    if (devs[dev] == CL_DEVICE_TYPE_CPU ) {
      global_work_size = compute_units * 1; // 1 thread per core
      local_work_size = 1;
      printf("Processing using the CPU with %d-threads-per-block ...\n", local_work_size);
    } else {
      cl_float ws = 64; // represents the warp (NVIDIA) or wavefront (ATI)
      global_work_size = compute_units * ws;
      while( (numOfItems / 4) % global_work_size != 0 ) global_work_size += ws;

      local_work_size = ws;
      printf("Processing using the GPU with %d-threads-per-block ...\n", local_work_size);
    }
    num_groups = global_work_size / local_work_size;

    context  = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cQFMA    = clCreateCommandQueue(context, device, 0, NULL);
    cQMAD    = clCreateCommandQueue(context, device, 0, NULL);
    if (cQMAD == NULL || cQFMA == NULL ) { 
      printf("Cannot create device, darn...exiting...\n");
      exit(-1);
    }
    const char* files[1] = {"fma_mad_cmp.cl"};
    const int NUMBER_OF_FILES = 1;
    char* programSource[NUMBER_OF_FILES];
    size_t sizes[NUMBER_OF_FILES];
    loadProgramSource(files, NUMBER_OF_FILES, programSource, sizes);
    program  = clCreateProgramWithSource(context, 1, (const char**)programSource, sizes, NULL);
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    mad  = clCreateKernel(program, "mad_test" , NULL);
    fma  = clCreateKernel(program, "fma_test" , NULL);

    // Prepare memory data structures
    a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                              numOfItems * sizeof(cl_float),
                              src_ptr, NULL);

    b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                              numOfItems * sizeof(cl_float),
                              src_ptr, NULL);

    c_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                              numOfItems * sizeof(cl_float),
                              src_ptr, NULL);

    fma_res_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, numOfItems * sizeof(cl_float),
                                    NULL, NULL);

    mad_res_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, numOfItems * sizeof(cl_float),
                                    NULL, NULL);

    fma_dst_ptr = (cl_float*) malloc(numOfItems * sizeof(cl_float));
    mad_dst_ptr = (cl_float*) malloc(numOfItems * sizeof(cl_float));

    // Set the necessary arguments to kernel's parameters
    clSetKernelArg(fma, 0, sizeof(cl_mem),  &a_buffer);
    clSetKernelArg(fma, 1, sizeof(cl_mem),  &b_buffer);
    clSetKernelArg(fma, 2, sizeof(cl_mem),  &c_buffer);
    clSetKernelArg(fma, 3, sizeof(cl_mem),  &fma_res_buffer);

    clSetKernelArg(mad, 0, sizeof(cl_mem),  &a_buffer);
    clSetKernelArg(mad, 1, sizeof(cl_mem),  &b_buffer);
    clSetKernelArg(mad, 2, sizeof(cl_mem),  &c_buffer);
    clSetKernelArg(mad, 3, sizeof(cl_mem),  &mad_res_buffer);
   
    global_work_size = DATA_SIZE ;
    clEnqueueNDRangeKernel(cQFMA, fma, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    clEnqueueNDRangeKernel(cQMAD, mad, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(cQFMA, fma_res_buffer, CL_TRUE, 0, numOfItems *sizeof(cl_float), fma_dst_ptr , 0, NULL, NULL);
    clEnqueueReadBuffer(cQMAD, mad_res_buffer, CL_TRUE, 0, numOfItems *sizeof(cl_float), mad_dst_ptr , 0, NULL, NULL);

    clFinish(cQFMA);
    clFinish(cQMAD);

    if (valuesOK(fma_dst_ptr, mad_dst_ptr, numOfItems))  
      printf("Check has passed!\n");
    else
      printf("Check has failed!\n");


    clReleaseMemObject(fma_res_buffer);
    clReleaseMemObject(mad_res_buffer);
    clReleaseMemObject(a_buffer);
    clReleaseMemObject(b_buffer);
    clReleaseMemObject(c_buffer);
    clReleaseCommandQueue(cQMAD);
    clReleaseCommandQueue(cQFMA);
    clReleaseProgram(program);
    free(fma_dst_ptr);
    free(mad_dst_ptr);
  }

  free(src_ptr);
}


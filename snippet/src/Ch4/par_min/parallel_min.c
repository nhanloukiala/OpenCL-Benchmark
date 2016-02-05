#ifdef APPLE
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <alloca.h>

#define NDEVS   2
#define DATA_SIZE 4096 * 4096;

/*
    This program should run on any device that supports OpenCL 1.1

    Note: In the event that it runs on the CPU, then we cannot assign
          a local_work_size > 1 since the CPU forbids it. In the GPU, 
          that's an entirely different story :) whew
*/
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


int main(int argc, char** argv) {
  cl_platform_id platform;
  int dev;
  cl_device_type devs[NDEVS] = { CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU };

  cl_uint *src_ptr;
  unsigned int numOfItems = DATA_SIZE;

  src_ptr = (cl_uint*) malloc( numOfItems * sizeof(cl_uint));
  cl_uint min = (cl_uint) -1;

  for( int i = 0; i < numOfItems; ++i) {
    src_ptr[i] = (cl_uint) i;
    min = src_ptr[i] < min? src_ptr[i] : min;
  }

  // Get the supported platforms
  clGetPlatformIDs(1, &platform, NULL);
  for( dev = 0; dev < NDEVS; dev++ ) { 
    cl_device_id     device;
    cl_context       context;
    cl_command_queue cQ;
    cl_program       program;
    cl_kernel        paraMin;
    cl_kernel        reduce;

    cl_mem           src_buffer;
    cl_mem           dst_buffer;
    
    cl_uint          *dst_ptr;
    
    clGetDeviceIDs(platform, devs[dev], 1, &device, NULL);

    cl_uint compute_units;
    size_t  global_work_size;
    size_t  local_work_size;
    size_t  num_groups;

    size_t  paramSize = 0;
    clGetDeviceInfo( device, CL_DEVICE_MAX_COMPUTE_UNITS, 0, NULL, &paramSize );
    cl_uint* ret = (cl_uint*) alloca(sizeof(cl_uint) * paramSize);
    clGetDeviceInfo( device, CL_DEVICE_MAX_COMPUTE_UNITS, paramSize, ret, NULL);
    compute_units = *ret;

    // Since the CPU doesn't permit parallel executing threads per core
    // hence we assign 1 thread to per core
    if (devs[dev] == CL_DEVICE_TYPE_CPU ) {
      global_work_size = compute_units * 1; // 1 thread per core
      local_work_size = 1;
    } else {
      cl_uint ws = 64; // represents the warp (NVIDIA) or wavefront (ATI)
      global_work_size = compute_units * ws;
      while( (numOfItems / 4) % global_work_size != 0 ) global_work_size += ws;

      local_work_size = ws;
    }
    num_groups = global_work_size / local_work_size;

    context  = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cQ       = clCreateCommandQueue(context, device, 0, NULL);
    if (cQ == NULL ) { 
      printf("Cannot create device, darn...exiting...\n");
      exit(-1);
    }
    const char* files[1] = {"par_min.cl"};
    const int NUMBER_OF_FILES = 1;
    char* programSource[NUMBER_OF_FILES];
    size_t sizes[NUMBER_OF_FILES];
    loadProgramSource(files, NUMBER_OF_FILES, programSource, sizes);
    program  = clCreateProgramWithSource(context, 1, (const char**)programSource, NULL, NULL);
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    paraMin  = clCreateKernel(program, "par_min", NULL);
    reduce   = clCreateKernel(program, "reduce" , NULL);

    printf("Kernels created!");

    // Prepare memory data structures
    src_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                numOfItems * sizeof(cl_uint),
                                src_ptr, NULL);
    dst_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, global_work_size * sizeof(cl_uint),
                                NULL, NULL);
    dst_ptr = (cl_uint*) malloc(numOfItems * sizeof(cl_uint));

    // Set the necessary arguments to kernel's parameters
    clSetKernelArg(paraMin, 0, sizeof(cl_mem),     &src_buffer);
    clSetKernelArg(paraMin, 1, sizeof(cl_mem),     &dst_buffer);
    clSetKernelArg(paraMin, 2, sizeof(cl_mem),     NULL);
    clSetKernelArg(paraMin, 3, sizeof(numOfItems), &numOfItems);
    clSetKernelArg(paraMin, 4, sizeof(dev),        &dev);
    
    clSetKernelArg(reduce, 0, sizeof(cl_mem),      &src_buffer);
    clSetKernelArg(reduce, 0, sizeof(cl_mem),      &dst_buffer);

    // Using events to manage execution dependency
    // so the kernel 'reduce' runs after 'paraMin'
    cl_event waitForParaMinToComplete;
    clEnqueueNDRangeKernel(cQ, paraMin, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &waitForParaMinToComplete);
    clEnqueueNDRangeKernel(cQ, reduce , 1, NULL, &num_groups, NULL, 1, &waitForParaMinToComplete, NULL);

    clFinish(cQ);

    clEnqueueReadBuffer(cQ, dst_buffer, CL_TRUE, 0, num_groups *sizeof(cl_uint),dst_ptr , 0, NULL, NULL);

    printf("computed min=%d, local min=%d\n", dst_ptr[0], min);
    if(dst_ptr[0] == min)
      printf("Check has passed!\n");
    else
      printf("Check has failed!\n");

    clReleaseMemObject(dst_buffer);
    clReleaseProgram(program);
    clReleaseCommandQueue(cQ);
    free(dst_ptr);
  }

  free(src_ptr);
}


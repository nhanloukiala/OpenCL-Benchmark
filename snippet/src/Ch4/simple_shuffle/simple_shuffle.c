#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define DATA_SIZE 1024
//#define DATA_SIZE 16 // for small runs
#define ITERATIONS 6

/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {

   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   } 

   /* Access a device */
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);   
   }

   return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err;

   /* Read program file and place content into buffer */
   program_handle = fopen(filename, "r");
   if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   /* Create program from file */
   program = clCreateProgramWithSource(ctx, 1, 
      (const char**)&program_buffer, &program_size, &err);
   if(err < 0) {
      perror("Couldn't create the program");
      exit(1);
   }
   free(program_buffer);

   /* Build program */
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if(err < 0) {

      /* Find size of log and print to std output */
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return program;
}

int main() {

   /* OpenCL data structures */
   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel kernel;
   cl_int i, err;

   /* Data and buffers */
   cl_float a_ptr[DATA_SIZE];
   cl_float b_ptr[DATA_SIZE];
   cl_int   mask[DATA_SIZE];
   cl_float res_ptr[DATA_SIZE];
   cl_mem a_buffer, b_buffer;
   cl_mem mask_buffer;
   cl_mem res_buffer;

   for(int i = 0; i < DATA_SIZE; ++i) {
        a_ptr[i] = i;
    }
   for(int i = 0, j = DATA_SIZE; i < DATA_SIZE; --j, ++i) {
        b_ptr[i] = j;
    }
   
   /* Create a context */
   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);   
   }

   /* Create a kernel by name */
   program = build_program(context, device, "simple_shuffle.cl");
   kernel = clCreateKernel(program, "permutate", &err);
   if(err < 0) {
      perror("Couldn't create a kernel");
      exit(1);   
   };

   a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*DATA_SIZE, a_ptr, &err);
   if(err < 0) {
      perror("Couldn't create buffer 'a'");
      exit(1);   
   };
   b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*DATA_SIZE, b_ptr, &err);
   if(err < 0) {
      perror("Couldn't create buffer 'b'");
      exit(1);   
   };
   /* Create a write-only buffer to hold the output data */
   res_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*DATA_SIZE, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);   
   };
        

   /* Create kernel argument */
   clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buffer);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buffer);
   clSetKernelArg(kernel, 3, sizeof(cl_mem), &res_buffer);
   
   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, 0, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
   };

   /* seed the random number generator */
   srandom(41L);
   for(int iter = 0; iter < ITERATIONS; ++iter) {
		   for(int i = 0; i < DATA_SIZE; ++i) {
		     mask[i] = random() % DATA_SIZE;
             #ifdef DEBUG
             printf("mask[%d]=%d\n", i, mask[i]);
             #endif
           }
		
           clSetKernelArg(kernel, 2, sizeof(cl_mem), &mask_buffer);
		   mask_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*DATA_SIZE, mask, &err);
		   if(err < 0) {
		      perror("Couldn't create a mask");
		      exit(1);   
		   };
		 
		   /* Enqueue kernel */
		   //err = clEnqueueTask(queue, kernel, 0, NULL, NULL);
           //size_t globalTs[1] = {DATA_SIZE };
           size_t globalTs[1] = {DATA_SIZE / 16};
           err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalTs, NULL, 0, NULL, NULL);
		   if(err < 0) {
		      perror("Couldn't enqueue the kernel");
		      exit(1);   
		   }
		
		   /* Read and print the result */
		   err = clEnqueueReadBuffer(queue, res_buffer, CL_TRUE, 0, sizeof(cl_float)*DATA_SIZE, &res_ptr, 0, NULL, NULL);
		   if(err < 0) {
		      perror("Couldn't read the buffer");
		      exit(1);   
		   }
		   
		   printf("\n\nshuffle: ");
		   for(i=0; i<DATA_SIZE; i++) {
		      printf("\t%.2f, ", res_ptr[i]);
		   }
		   printf("\n");
           clReleaseMemObject(mask_buffer);
   }
   /* Deallocate resources */
   clReleaseMemObject(a_buffer);
   clReleaseMemObject(b_buffer);   
   clReleaseMemObject(res_buffer);   
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}

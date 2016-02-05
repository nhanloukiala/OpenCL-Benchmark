#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <alloca.h>
#include "user_buffer.h"

#ifdef APPLE
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define DATA_SIZE 1048576

// test for valid values
int valuesOK(UserData* to, UserData* from) {
    for(int i = 0; i < DATA_SIZE; ++i) {
        if ( to[i].w != from[i].w ) return 0;
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

int main(int argc, char** argv) {

   /* OpenCL 1.1 data structures */
   cl_platform_id* platforms;
   cl_program program;
   cl_device_id device;
   cl_context context;

   /* OpenCL 1.1 scalar data types */
   cl_uint numOfPlatforms;
   cl_int  error;

   /*
    Prepare an array of UserData via dynamic memory allocation
   */
   UserData* ud_in = (UserData*) malloc( sizeof(UserData) * DATA_SIZE); // input to device
   UserData* ud_out = (UserData*) malloc( sizeof(UserData) * DATA_SIZE); // output from device
   for( int i = 0; i < DATA_SIZE; ++i) {
      (ud_in + i)->x = i;
      (ud_in + i)->y = i;
      (ud_in + i)->z = i;
      (ud_in + i)->w = 3 * i;
   }
   /* 
      Get the number of platforms 
      Remember that for each vendor's SDK installed on the computer,
      the number of available platform also increased. 
    */
   error = clGetPlatformIDs(0, NULL, &numOfPlatforms);
   if(error != CL_SUCCESS ) {			
      perror("Unable to find any OpenCL platforms");
      exit(1);
   }

   platforms = (cl_platform_id*) alloca(sizeof(cl_platform_id) * numOfPlatforms);
   printf("Number of OpenCL platforms found: %d\n", numOfPlatforms);

   error = clGetPlatformIDs(numOfPlatforms, platforms, NULL);
   if(error != CL_SUCCESS ) {			
      perror("Unable to find any OpenCL platforms");
      exit(1);
   }
   // Search for a CPU/GPU device through the installed platforms
   // Build a OpenCL program and do not run it.
   for(cl_uint i = 0; i < numOfPlatforms; i++ ) {
       // Get the GPU device
       error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
       if(error != CL_SUCCESS) {
          // Otherwise, get the CPU
          error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 1, &device, NULL);
       }
        if(error != CL_SUCCESS) {
            perror("Can't locate any OpenCL compliant device");
            exit(1);
        }
        /* Create a context */
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
        if(error != CL_SUCCESS) {
            perror("Can't create a valid OpenCL context");
            exit(1);
        }

        /* Load the two source files into temporary datastores */
        const char *file_names[] = {"user_test.cl"}; 
        const int NUMBER_OF_FILES = 1;
        char* buffer[NUMBER_OF_FILES];
        size_t sizes[NUMBER_OF_FILES];
        loadProgramSource(file_names, NUMBER_OF_FILES, buffer, sizes);

        /* Create the OpenCL program object */
        program = clCreateProgramWithSource(context, NUMBER_OF_FILES, (const char**)buffer, sizes, &error);				
	    if(error != CL_SUCCESS) {
	      perror("Can't create the OpenCL program object");
	      exit(1);   
	    }
        /* Build OpenCL program object and dump the error message, if any */
        char *program_log;
        size_t log_size;
        error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);		
	    if(error != CL_SUCCESS) {
	      // If there's an error whilst building the program, dump the log
	      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	      program_log = (char*) malloc(log_size+1);
	      program_log[log_size] = '\0';
	      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
	            log_size+1, program_log, NULL);
	      printf("\n=== ERROR ===\n\n%s\n=============\n", program_log);
	      free(program_log);
	      exit(1);
	    }
  
        /* Query the program as to how many kernels were detected */
        cl_uint numOfKernels;
        error = clCreateKernelsInProgram(program, 0, NULL, &numOfKernels);
        if (error != CL_SUCCESS) {
            perror("Unable to retrieve kernel count from program");
            exit(1);
        }
        cl_kernel* kernels = (cl_kernel*) alloca(sizeof(cl_kernel) * numOfKernels);
        error = clCreateKernelsInProgram(program, numOfKernels, kernels, NULL);
        for(cl_uint i = 0; i < numOfKernels; i++) {
            char kernelName[32];
            cl_uint argCnt;
            clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME, sizeof(kernelName), kernelName, NULL);
            clGetKernelInfo(kernels[i], CL_KERNEL_NUM_ARGS, sizeof(argCnt), &argCnt, NULL);
            printf("Kernel name: %s with arity: %d\n", kernelName, argCnt);
            printf("About to create command queue and enqueue this kernel...\n");

            /* Create a command queue */
            cl_command_queue cQ = clCreateCommandQueue(context, device, 0, &error);
            if (error != CL_SUCCESS) { 
                perror("Unable to create command-queue");
                exit(1);
            }

            /* Create a OpenCL buffer object */
            cl_mem UDObj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                           sizeof(UserData) * DATA_SIZE, ud_in, &error);
            if (error != CL_SUCCESS) { 
                perror("Unable to create buffer object");
                exit(1);
            }

            /* Let OpenCL know that the kernel is suppose to receive an argument */
            error = clSetKernelArg(kernels[i], 0, sizeof(cl_mem), &UDObj);
            if (error != CL_SUCCESS) { 
                perror("Unable to create buffer object");
                exit(1);
            }

            /* Enqueue the kernel to the command queue */
            error = clEnqueueTask(cQ, kernels[i], 0, NULL, NULL);
            if (error != CL_SUCCESS) { 
                perror("Unable to enqueue task to command-queue");
                exit(1);
            }
            printf("Task has been enqueued successfully!\n");

            /* Enqueue the read-back from device to host */
            error = clEnqueueReadBuffer(cQ, UDObj,
                                         CL_TRUE,                    // blocking read
                                         0,                          // write from the start
                                         sizeof(UserData) * DATA_SIZE, // how much to copy
                                         ud_out, 0, NULL, NULL);
            if ( valuesOK(ud_in, ud_out) ) {
                printf("Check passed!\n");
            } else printf("Check failed!\n");

            /* Release the command queue */
            clReleaseCommandQueue(cQ);
            clReleaseMemObject(UDObj);
        }

        /* Clean up */
        
        for(cl_uint i = 0; i < numOfKernels; i++) { clReleaseKernel(kernels[i]); }
        for(i=0; i< NUMBER_OF_FILES; i++) { free(buffer[i]); }
        clReleaseProgram(program);
        clReleaseContext(context);
   }

   free(ud_in);
   free(ud_out);
}

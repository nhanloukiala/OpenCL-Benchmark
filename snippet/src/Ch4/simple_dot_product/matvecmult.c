#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <alloca.h>
#include "matvecmult_config.h"

#ifdef APPLE
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define VECTOR_LENGTH 4
//#define DATA_SIZE 16      // for test runs,
#define DATA_SIZE 1048576 // for standard runs,
//#define DATA_SIZE 2097152   // for large runs,

/*
    This program requires all the devices to be supported by the
    OpenCL 1.1 Refer to the Khronos Group for list of supported
    vendors.

    The program will partition the data against all detected devices
    and execute on them. In the setup i've got, i have a ATI 6870x2 GPU card
    and a Intel Core i7 CPU.
*/

//
// The test for valid values is a little interesting and its
// not what OpenCL 1.1 suggests. You should be able to say 
// to[i].s0, to[i].s1 etc but turns out you cannot do that because of
// the way AMD SDK has defined it in "cl_platform.h" slightly differently
//
int valuesOK(cl_float* from, cl_float* to, cl_float* vec,  size_t length) {
#ifdef DEBUG
    printf("Checking data of size: %lu\n", length);
#endif

    for(int i = 0; i < length; ++i ) {
        float fromDP = 0.0f;
        fromDP += from[i] * vec[0];
        fromDP += from[i*4 + 1] * vec[1];
        fromDP += from[i*4 + 2] * vec[2];
        fromDP += from[i*4 + 3] * vec[3];
#ifdef DEBUG
        printf("local[%i]=%f vs gpu = %f\n", i, fromDP, to[i]);
        if (fromDP != to[i]) return 0;
#endif
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
   cl_context context;

   /* OpenCL 1.1 scalar data types */
   cl_uint numOfPlatforms;
   cl_int  error;

   cl_float* ud_in = (cl_float*) malloc( sizeof(cl_float) * DATA_SIZE); // input to device
   cl_float* vec_in = (cl_float*) malloc( sizeof(cl_float) * VECTOR_LENGTH); 
   cl_float* ud_out = (cl_float*) malloc( sizeof(cl_float) * DATA_SIZE); // output from device
   for( int i = 0; i < DATA_SIZE; ++i) {
       ud_in[i] = (cl_float){ (float)i };
   }

   for( int i = 0; i < VECTOR_LENGTH; ++i ) 
       vec_in[i] = i * 3.0f;

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

        cl_uint numOfDevices = 0;

        /* Determine how many devices are connected to your platform */
        error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numOfDevices);
        if (error != CL_SUCCESS ) { 
            perror("Unable to obtain any OpenCL compliant device info");
            exit(1);
        }
        cl_device_id* devices = (cl_device_id*) alloca(sizeof(cl_device_id) * numOfDevices);

        /* Load the information about your devices into the variable 'devices' */
        error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numOfDevices, devices, NULL);
        if (error != CL_SUCCESS ) { 
            perror("Unable to obtain any OpenCL compliant device info");
            exit(1);
        }
        printf("Number of detected OpenCL devices: %d\n", numOfDevices);

	    /* Create a context */
        cl_context_properties ctx[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i], 0 };
	    context = clCreateContext(ctx, numOfDevices, devices, NULL, NULL, &error);
	    if(error != CL_SUCCESS) {
	        perror("Can't create a valid OpenCL context");
	        exit(1);
	    }

	    /* For each device, create a buffer and partition that data among the devices for compute! */
	    cl_mem inobj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
	                                  sizeof(float) * DATA_SIZE, ud_in, &error);
	    if(error != CL_SUCCESS) {
	        perror("Can't create a buffer");
	        exit(1);
	    }

        int offset = 0; 
        for(int i = 0; i < numOfDevices; ++i, ++offset ) {
	        /* Load the two source files into temporary datastores */
	        const char *file_names[] = {"matvecmult.cl"}; 
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
	        error = clBuildProgram(program, 1, &devices[i], NULL, NULL, NULL);		
		    if(error != CL_SUCCESS) {
		      // If there's an error whilst building the program, dump the log
		      clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		      program_log = (char*) malloc(log_size+1);
		      program_log[log_size] = '\0';
		      clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG, 
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

            /* Loop thru each kernel and execute on device */
	        for(cl_uint j = 0; j < numOfKernels; j++) {
	            char kernelName[32];
	            cl_uint argCnt;
	            clGetKernelInfo(kernels[j], CL_KERNEL_FUNCTION_NAME, sizeof(kernelName), kernelName, NULL);
	            clGetKernelInfo(kernels[j], CL_KERNEL_NUM_ARGS, sizeof(argCnt), &argCnt, NULL);
	            printf("Kernel name: %s with arity: %d\n", kernelName, argCnt);
	            printf("About to create command queue and enqueue this kernel...\n");
	
	            /* Create a command queue */
	            cl_command_queue cQ = clCreateCommandQueue(context, devices[i], 0, &error);
	            if (error != CL_SUCCESS) { 
	                perror("Unable to create command-queue");
	                exit(1);
	            }
	
                /* Create a buffer and copy the data from the main buffer */
	            cl_mem outobj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
	                                           sizeof(float) * DATA_SIZE, 0, &error);
	            if (error != CL_SUCCESS) { 
	                perror("Unable to create output-buffer object");
	                exit(1);
	            }

                /* Create a vector buffer */
	            cl_mem vecobj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
	                                           sizeof(float) * VECTOR_LENGTH, vec_in, &error);
	            if (error != CL_SUCCESS) { 
	                perror("Unable to create vec-buffer object");
                    exit(1);
	            }
                
	            /* Let OpenCL know that the kernel is suppose to receive an argument */
	            error = clSetKernelArg(kernels[j], 0, sizeof(cl_mem), &inobj);
	            error = clSetKernelArg(kernels[j], 1, sizeof(cl_mem), &vecobj);
	            error = clSetKernelArg(kernels[j], 2, sizeof(cl_mem), &outobj);
	            if (error != CL_SUCCESS) { 
	                perror("Unable to set buffer object in kernel");
	                exit(1);
	            }
	
	            /* Enqueue the kernel to the command queue */
                size_t threadsPerGroup[] = {4};  // 4 work-items per kernel
                size_t numOfGroups = DATA_SIZE / threadsPerGroup[0] ; // X work-groups 
                error = clEnqueueNDRangeKernel(cQ,
                                               kernels[j],
                                               1,
                                               NULL,
                                               &numOfGroups,
                                               NULL,0, NULL, NULL);
	            if (error != CL_SUCCESS) { 
	                perror("Unable to enqueue task to command-queue");
	                exit(1);
	            }
	            printf("Task has been enqueued successfully!\n");

	            /* Enqueue the read-back from device to host */
	            error = clEnqueueReadBuffer(cQ, outobj,
	                                        CL_TRUE,               // blocking read
	                                        0,                      // read from the start
	                                        sizeof(cl_float)*DATA_SIZE,          // how much to copy
	                                        ud_out, 0, NULL, NULL);
                /* Check the returned data */
	            if ( valuesOK(ud_in, ud_out, vec_in, DATA_SIZE/4) ) {
	                printf("Check passed!\n");
	            } else printf("Check failed!\n");
	
	            /* Release the command queue */
	            clReleaseCommandQueue(cQ);
	            clReleaseMemObject(outobj);
	        } 

        /* Clean up */
        
        for(cl_uint i = 0; i < numOfKernels; i++) { clReleaseKernel(kernels[i]); }
        for(int i=0; i< NUMBER_OF_FILES; i++) { free(buffer[i]); }
        clReleaseProgram(program);
    }// end of device loop and execution
    
	    clReleaseMemObject(inobj);
        clReleaseContext(context);
   }// end of platform loop

   free(ud_in);
   free(ud_out);
}

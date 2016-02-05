#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <alloca.h>
#include <math.h>

#ifdef APPLE
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

//#define DATA_SIZE 64      // for test runs,
#define DATA_SIZE 1048576 // for standard runs,
//#define DATA_SIZE 8388608   // for large runs,

/*
    This program requires all the devices to be supported by the
    OpenCL 1.1 Refer to the Khronos Group for list of supported
    vendors.

    The program will partition the data against all detected devices
    and execute on them 'clEnqueueNDRange'. In the setup i've got, 
    i have a ATI 6870x2 GPU card and a Intel Core i7 CPU.
*/

// test for valid values
int valuesOK(float* to, float* from, size_t length) {
#ifdef DEBUG
    printf("Checking data of size: %lu\n", length);
#endif
    for(int i = 0; i < length; ++i) {
#ifdef DEBUG
        printf("to:%f, from:%f\n", to[i] ,from[i]);
#endif
        if ( to[i] != from[i]) return 0;
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

void displayDeviceType(cl_device_id id,
                          cl_device_info param_name) {
  cl_int error = 0;
  size_t paramSize = 0;

  error = clGetDeviceInfo( id, param_name, 0, NULL, &paramSize );
  if (error != CL_SUCCESS ) {
    perror("Unable to obtain device info for param\n");
    return;
  }

  /* the cl_device_info are preprocessor directives defined in cl.h */
  switch (param_name) {
    case CL_DEVICE_TYPE: {
            cl_device_type* devType = (cl_device_type*) alloca(sizeof(cl_device_type) * paramSize);
            error = clGetDeviceInfo( id, param_name, paramSize, devType, NULL );
            if (error != CL_SUCCESS ) {
                perror("Unable to obtain device info for param\n");
                return;
            }
            switch (*devType) {
              case CL_DEVICE_TYPE_CPU : printf("Running on CPU ........\n");break;
              case CL_DEVICE_TYPE_GPU : printf("Running GPU ........\n");break;
              case CL_DEVICE_TYPE_ACCELERATOR : printf("Accelerator detected\n");break;
              case CL_DEVICE_TYPE_DEFAULT : printf("default detected\n");break;
            }
    }break;
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

   /*
    Prepare an array of UserData via dynamic memory allocation
   */
   cl_float* h_in = (float*) malloc( sizeof(cl_float4) * DATA_SIZE); // input to device
   cl_float* h_out = (float*) malloc( sizeof(cl_float4) * DATA_SIZE); // output from device
   for( int i = 0; i < DATA_SIZE; ++i) {
        h_in[i] = (float)i;
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
	    cl_mem memInObj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
	                                     sizeof(cl_float4) * (DATA_SIZE), h_in, &error);
	    if(error != CL_SUCCESS) {
	        perror("Can't create an input buffer object");
	        exit(1);
	    }

        int offset = 0; 
        for(int i = 0; i < numOfDevices; ++i, ++offset ) {

            displayDeviceType(devices[i], CL_DEVICE_TYPE);

	        /* Load the two source files into temporary datastores */
	        const char *file_names[] = {"work_partition.cl"}; 
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
	            printf("\t=> Kernel name: %s with arity: %d\n", kernelName, argCnt);
	            printf("\t=> About to create command queue and enqueue this kernel...\n");
	
	            /* Create a command queue */
	            cl_command_queue cQ = clCreateCommandQueue(context, devices[i], 0, &error);
	            if (error != CL_SUCCESS) { 
	                perror("Unable to create command-queue");
	                exit(1);
	            }
	
	            cl_mem memOutObj = clCreateBuffer(context, CL_MEM_WRITE_ONLY ,
	                                              sizeof(cl_float4) * (DATA_SIZE), NULL, &error);
	            if(error != CL_SUCCESS) {
	                perror("Can't create an output buffer object");
	                exit(1);
	            }

	            /* Let OpenCL know that the kernel is suppose to receive two arguments */
	            error = clSetKernelArg(kernels[j], 0, sizeof(cl_mem), &memInObj);
	            if (error != CL_SUCCESS) { 
	                perror("Unable to set buffer object in kernel");
	                exit(1);
	            }
	            error = clSetKernelArg(kernels[j], 1, sizeof(cl_mem), &memOutObj);
	            if (error != CL_SUCCESS) { 
	                perror("Unable to set buffer object in kernel");
	                exit(1);
	            }
	
	            /* Enqueue the kernel to the command queue */
                size_t globalThreads[2];
                globalThreads[0]=1024;
                globalThreads[1]=1024;
                size_t localThreads[2];
                localThreads[0] = 64;
                localThreads[1] = 2;
	
                cl_event evt;
	            error = clEnqueueNDRangeKernel(cQ, 
                                               kernels[j],
                                               2,
                                               0,
                                               globalThreads,
                                               localThreads,
                                               0, 
                                               NULL, &evt);
                clWaitForEvents(1, &evt);
	            if (error != CL_SUCCESS) { 
	                perror("Unable to enqueue task to command-queue");
	                exit(1);
	            }
                clReleaseEvent(evt);
	            printf("\t=> Task has been enqueued successfully!\n");
	
	            /* Enqueue the read-back from device to host */
	            error = clEnqueueReadBuffer(cQ, memOutObj,
	                                        CL_TRUE,               // blocking read
	                                        0,         // write from the last offset
	                                        (DATA_SIZE)*sizeof(cl_float4),           // how much to copy
	                                        h_out, 0, NULL, NULL);

                /* Check the returned data */
	            if ( valuesOK(h_in, h_out, DATA_SIZE) ) {
	                printf("Check passed!\n");
	            } else printf("Check failed!\n");
	
	            /* Release the resources */
	            clReleaseCommandQueue(cQ);
	            clReleaseMemObject(memOutObj);
	        } 

        /* Clean up */
        
        for(cl_uint i = 0; i < numOfKernels; i++) { clReleaseKernel(kernels[i]); }
        for(int i=0; i< NUMBER_OF_FILES; i++) { free(buffer[i]); }
        clReleaseProgram(program);
    }// end of device loop and execution
    
	    clReleaseMemObject(memInObj);
        clReleaseContext(context);
   }// end of platform loop

   free(h_in);
   free(h_out);
}

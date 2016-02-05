#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <alloca.h>
#include <string.h>

#ifdef APPLE
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define BLOCK_SIZE 256
//#define DATA_SIZE 1048576   // for standard runs,
#define DATA_SIZE 4194304   // for average runs,
//#define DATA_SIZE 8388608   // for large runs,

int valuesOK(unsigned long hostSum, 
             unsigned int numOfBlocks,
             cl_int* deviceData) {
    unsigned long deviceSum = 0;
    for(int i = 0; i < numOfBlocks; ++i) 
        deviceSum += deviceData[i];
    //printf("hostSum = %ul, deviceSum = %ul\n", hostSum, deviceSum);
    if(hostSum == deviceSum) return 1; else return 0;
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

   /*
    Prepare the input and output  
   */
   unsigned int numOfBlocks = DATA_SIZE / (BLOCK_SIZE * 2);
   unsigned long refSum = 0L;

   cl_int* input = (cl_int*) malloc( sizeof(cl_int) * DATA_SIZE); // input to device
   cl_int* output = (cl_int*) malloc( sizeof(cl_int) * numOfBlocks); // output from device
   for( int i = 1; i <= DATA_SIZE; ++i) {
        input[i-1] = i;
        refSum += i;
   }
   memset(output, 0, numOfBlocks);

   // we're assuming that both DATA_SIZE and BLOCK_SIZE are powers of 2
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

        /* We're only looking for the GPUs installed on the machine */
        error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &numOfDevices);
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

	    cl_mem inputObj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
	                                     sizeof(cl_int) * DATA_SIZE, input, &error);
	    if(error != CL_SUCCESS) {
	        perror("Can't create a buffer");
	        exit(1);
	    }

	    cl_mem outputObj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
	                                      sizeof(cl_int) * numOfBlocks, output, &error);
        int offset = 0; 
        for(int i = 0; i < numOfDevices; ++i, ++offset ) {
	        /* Load the two source files into temporary datastores */
	        const char *file_names[] = {"reduction.cl"}; 
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
	            cl_command_queue commandQueue = clCreateCommandQueue(context, devices[i], 0, &error);
	            if (error != CL_SUCCESS) { 
	                perror("Unable to create command-queue");
	                exit(1);
	            }
	
	            /* Let OpenCL know that the kernel is suppose to receive an argument */
	            error = clSetKernelArg(kernels[j], 0, sizeof(cl_mem), (void*)&inputObj);
	            error = clSetKernelArg(kernels[j], 1, sizeof(cl_mem), (void*)&outputObj);
	            error = clSetKernelArg(kernels[j], 2, BLOCK_SIZE * sizeof(cl_int), NULL);
	            if (error != CL_SUCCESS) { 
	                perror("Unable to set buffer object in kernel");
	                exit(1);
	            }
	

                size_t globalThreads[] = {DATA_SIZE};
                size_t localThreads[] = {BLOCK_SIZE};
                cl_event profileEvt;
			    error  = clEnqueueNDRangeKernel(commandQueue,
			                                    kernels[j],
			                                    1,  
			                                    NULL,
			                                    globalThreads,
			                                    localThreads,
			                                    0,  
			                                    NULL,
			                                    &profileEvt);
                clFinish(commandQueue);

                // Profile the kernel
                cl_ulong startTime, endTime;
                clGetEventProfilingInfo(profileEvt, CL_PROFILING_COMMAND_START, sizeof(startTime), &startTime, NULL);
                clGetEventProfilingInfo(profileEvt, CL_PROFILING_COMMAND_START, sizeof(endTime), &endTime, NULL);
                printf("Bandwidth is %f GB per second\n",((endTime - startTime)/1e9));
	            /* read-back from device to host and we override the original input */
	            error = clEnqueueReadBuffer(commandQueue, outputObj,
	                                        CL_TRUE,               // blocking read
	                                        0,                      // read from the start
	                                        sizeof(cl_int)*numOfBlocks,          // how much to copy
	                                        output, 0, NULL, NULL);
                /* Check the returned data */
	            if ( valuesOK(refSum, numOfBlocks, output) ) {
	                printf("Check passed!\n");
	            } else printf("Check failed!\n");
	
	            /* Release the command queue */
	            clReleaseCommandQueue(commandQueue);
	        } 

        /* Clean up */
        
        for(cl_uint i = 0; i < numOfKernels; i++) { clReleaseKernel(kernels[i]); }
        for(int i=0; i< NUMBER_OF_FILES; i++) { free(buffer[i]); }
        clReleaseProgram(program);
    }// end of device loop and execution
    
	    clReleaseMemObject(inputObj);
	    clReleaseMemObject(outputObj);
        clReleaseContext(context);
   }// end of platform loop

   free(input);
   free(output);
}

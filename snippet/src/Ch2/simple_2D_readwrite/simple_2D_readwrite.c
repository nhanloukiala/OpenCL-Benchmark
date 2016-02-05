#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <alloca.h>

#ifdef APPLE
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define NUM_BUFFER_ELEMENTS 16

/*
    This program requires all the devices to be supported by the
    OpenCL 1.1 Refer to the Khronos Group for list of supported
    vendors.
*/


// test for valid values
int valuesOK(cl_int* to, cl_int* from, size_t length) {
#ifdef DEBUG
    printf("Checking data of size: %lu\n", length);
#endif
    for(int i = 0; i < length; ++i) {
#ifdef DEBUG
        printf("to:%d, from:%d\n", to[i] ,from[i]);
#endif
        if ( to[i] != from[i] ) return 0;
    }
  return 1;
}

int main(int argc, char** argv) {

   /* OpenCL 1.1 data structures */
   cl_platform_id* platforms;
   cl_context context;

   /* OpenCL 1.1 scalar data types */
   cl_uint numOfPlatforms;
   cl_int  error;

   cl_int hostBuffer[NUM_BUFFER_ELEMENTS] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
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
	    cl_mem UDObj = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR, 
	                                  sizeof(int) * NUM_BUFFER_ELEMENTS, hostBuffer, &error);
	    if(error != CL_SUCCESS) {
	        perror("Can't create a buffer");
	        exit(1);
	    }

        for(int i = 0; i < numOfDevices; ++i) {
	
	            /* Create a command queue */
	            cl_command_queue cQ = clCreateCommandQueue(context, devices[i], 0, &error);
	            if (error != CL_SUCCESS) { 
	                perror("Unable to create command-queue");
	                exit(1);
	            }

                cl_int outputPtr[16] = {-1, -1, -1, -1,-1, -1, -1, -1,-1, -1, -1, -1,-1, -1, -1, -1};
	            for(int idx = 0; idx < 4; ++ idx) {	
	                size_t buffer_origin[3] = {idx*2*sizeof(int), idx, 0}; 
	                size_t host_origin[3] = {idx*2*sizeof(int), idx, 0}; 
	                size_t region[3] = {2*sizeof(int), 2, 1};
	
		            /* Enqueue the read-back from device to host */
		            error = clEnqueueReadBufferRect(
	                       cQ,
	                       UDObj,
		                   CL_TRUE,               // blocking read
	                       buffer_origin,         
	                       host_origin,
	                       region,
	                       0, // buffer_row_pitch
	                       0, // buffer_slice_pitch
	                       0, // host_row_pitch
	                       0, // host_slice_pitch
	                       outputPtr, 0, NULL, NULL);
	
	            }

            #ifdef DEBUG
	            for(int i = 0; i < 16; i++) printf("%d\n", outputPtr[i]); 
            #endif
            if (valuesOK(hostBuffer, outputPtr,16)) printf("Check passed!\n"); else printf("Check failed!\n");
	            /* Release the command queue */
	            clReleaseCommandQueue(cQ);

        /* Clean up */
        
    }// end of device loop and execution
	    clReleaseMemObject(UDObj);
        clReleaseContext(context);
   }// end of platform loop

}//end of main

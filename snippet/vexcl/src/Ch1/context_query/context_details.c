#include <stdio.h>
#include <stdlib.h>
#include <alloca.h>

#ifdef APPLE
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


void displayPlatformInfo(cl_platform_id id,
                         cl_platform_info param_name,
                         const char* paramNameAsStr) {
    cl_int error = 0;
    size_t paramSize = 0;
    error = clGetPlatformInfo( id, param_name, 0, NULL, &paramSize );
    char* moreInfo = (char*)alloca( sizeof(char) * paramSize);
    error = clGetPlatformInfo( id, param_name, paramSize, moreInfo, NULL );
    if (error != CL_SUCCESS ) {
        perror("Unable to find any OpenCL platform information");
        return;
    }
    printf("%s: %s\n", paramNameAsStr, moreInfo);
}

void createAndReleaseContext(cl_platform_id id, 
                             cl_device_type dev_type) {
    /* OpenCL 1.1 device types */
    cl_int error = 0;
    cl_uint numOfDevices = 0;

    /* Determine how many devices are connected to your platform */
    error = clGetDeviceIDs(id, dev_type, 0, NULL, &numOfDevices);
    if (error != CL_SUCCESS ) { 
        perror("Unable to obtain any OpenCL compliant device info");
        exit(1);
    }
    cl_device_id* devices = (cl_device_id*) alloca(sizeof(cl_device_id) * numOfDevices);

    /* Load the information about your devices into the variable 'devices' */
    error = clGetDeviceIDs(id, dev_type, numOfDevices, devices, NULL);
    if (error != CL_SUCCESS ) { 
        perror("Unable to obtain any OpenCL compliant device info");
        exit(1);
    }
    printf("Number of detected OpenCL devices: %d\n", numOfDevices);

    /* 
       We attempt to create contexts for each device we find, report it
       and release the context. Once a context is created, its context is implicitly
       retained and so you don't have to invoke 'clRetainContext'
     */
    for(int i = 0; i < numOfDevices; ++ i ) {
        cl_context context = clCreateContext(NULL, 1, &devices[i], NULL, NULL, &error); 
        cl_uint ref_cnt = 0;
        if (error != CL_SUCCESS) {
            perror("Can't create a context");
            exit(1);
        }
        error = clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT, sizeof(ref_cnt), &ref_cnt, NULL);
        if (error != CL_SUCCESS) {
            perror("Can't obtain context information");
            exit(1);
        }
        printf("Reference count of device is %d\n", ref_cnt);

        // Release the context
        clReleaseContext(context);
    }
}

int main() {

   /* OpenCL 1.1 data structures */
   cl_platform_id* platforms;

   /* OpenCL 1.1 scalar data types */
   cl_uint numOfPlatforms;
   cl_int  error;

   /* 
      Get the number of platforms 
      Remember that for each vendor's SDK installed on the computer,
      the number of available platform also increased. 
    */
   error = clGetPlatformIDs(0, NULL, &numOfPlatforms);
   if(error != CL_SUCCESS) {			
      perror("Unable to find any OpenCL platforms");
      exit(1);
   }

   // Allocate memory for the number of installed platforms.
   // alloca(...) occupies some stack space but is automatically freed on return
   platforms = (cl_platform_id*) alloca(sizeof(cl_platform_id) * numOfPlatforms);
   printf("Number of OpenCL platforms found: %d\n", numOfPlatforms);

   error = clGetPlatformIDs(numOfPlatforms, platforms, NULL);
   if(error != CL_SUCCESS) {			
      perror("Unable to find any OpenCL platforms");
      exit(1);
   }
   // We invoke the API 'clPlatformInfo' twice for each parameter we're trying to extract
   // and we use the return value to create temporary data structures (on the stack) to store
   // the returned information on the second invocation.
   for(cl_uint i = 0; i < numOfPlatforms; ++i) {
        // Assume that we don't know how many devices are OpenCL compliant, we locate everything !
        createAndReleaseContext( platforms[i], CL_DEVICE_TYPE_ALL );
   }

   return 0;
}

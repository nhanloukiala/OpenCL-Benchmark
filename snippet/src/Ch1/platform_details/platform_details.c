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
        displayPlatformInfo( platforms[i], CL_PLATFORM_PROFILE, "CL_PLATFORM_PROFILE" );
        displayPlatformInfo( platforms[i], CL_PLATFORM_VERSION, "CL_PLATFORM_VERSION" );
        displayPlatformInfo( platforms[i], CL_PLATFORM_NAME,    "CL_PLATFORM_NAME" );
        displayPlatformInfo( platforms[i], CL_PLATFORM_VENDOR,  "CL_PLATFORM_VENDOR" );
        displayPlatformInfo( platforms[i], CL_PLATFORM_EXTENSIONS, "CL_PLATFORM_EXTENSIONS" );
   }

   return 0;
}

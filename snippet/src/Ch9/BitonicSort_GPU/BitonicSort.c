#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <alloca.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include "bitonicsort_config.h"

#ifdef  __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define GROUP_SIZE 256                   // ATI HD7870 has 20 parallel compute units
const unsigned int LENGTH = 1<<24;       
const unsigned int _SHARED_MEM_ = GROUP_SIZE * sizeof(cl_uint); // Size of shared memory on the GPU/CPU device (CPU doesn't matter)

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

int
decipherEvent(cl_event* event) {
    cl_int status = CL_SUCCESS;
    cl_int eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE) {
        status = clGetEventInfo(*event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &eventStatus, NULL);
    }
    clReleaseEvent(*event);
    return 0;
}

void fillRandom(int* data, unsigned int length, unsigned int seed) {
    int* iptr = data;
    
    if(!seed) seed = (unsigned int) time(NULL);
    srand(seed);
    for(int i = 0 ; i < length; ++i) 
            iptr[i] = rand() % 255;
}

int main(int argc, char** argv) {
    /* OpenCL 1.1 data structures */
    cl_platform_id* platforms;
    cl_program program;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_uint numOfPlatforms;
    cl_int  error;

    cl_mem  device_A_in = NULL;    // input buffer mem buffer
    cl_int* host_A_in = NULL;      // input buffer A
    cl_int* host_A_out = NULL;     // output buffer A

	{
	    host_A_in  = (cl_int*)malloc(LENGTH * sizeof(cl_int));
	    host_A_out = (cl_int*)malloc(LENGTH * sizeof(cl_int));

	    memset(host_A_in, 0, LENGTH * sizeof(cl_int));
	    memset(host_A_out, 0, LENGTH * sizeof(cl_int));
        
        fillRandom(host_A_in, LENGTH, 42);
    }

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
    
    platforms = (cl_platform_id*) alloca(sizeof(cl_platform_id) * numOfPlatforms);
    printf("Number of OpenCL platforms found: %d\n", numOfPlatforms);
    
    error = clGetPlatformIDs(numOfPlatforms, platforms, NULL);
    if(error != CL_SUCCESS) {
        perror("Unable to find any OpenCL platforms");
        exit(1);
    }
    // Search for a GPU device through the installed platforms
    // Build a OpenCL program and do not run it.
    for(cl_int i = 0; i < numOfPlatforms; i++ ) {
        // Get the GPU device
        error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);

        if(error != CL_SUCCESS) {
            perror("Can't locate a OpenCL compliant device i.e. GPU");
            exit(1);
        }
        /* Create a context */
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
        if(error != CL_SUCCESS) {
            perror("Can't create a valid OpenCL context");
            exit(1);
        }
        
        /* Load the two source files into temporary datastores */
        const char *file_names[] = {"BitonicSort.cl"};
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
        const char options[] = "";
        size_t log_size;

        error = clBuildProgram(program, 1, &device, options, NULL, NULL);
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
       
        // Queue is created with profiling enabled 
        cl_command_queue_properties props;
        props |= CL_QUEUE_PROFILING_ENABLE;

        queue = clCreateCommandQueue(context, device, props, &error);

#ifdef USE_SHARED_MEM
        cl_kernel kernel = clCreateKernel(program, "bitonicSort_sharedmem", &error);
#elif def USE_SHARED_MEM_2
        cl_kernel kernel = clCreateKernel(program, "bitonicSort_sharedmem_2", &error);
#else
        cl_kernel kernel = clCreateKernel(program, "bitonicSort", &error);
#endif
        device_A_in = clCreateBuffer(context,
                                     CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
                                     LENGTH * sizeof(cl_int),
                                     host_A_in,
                                     &error);

        cl_uint sortOrder = 0; // descending order else 1 for ascending order
        cl_uint stages = 0;
        for(unsigned int i = LENGTH; i > 1; i >>= 1) 
            ++stages;
        clSetKernelArg(kernel, 0, sizeof(cl_mem),(void*)&device_A_in);
        clSetKernelArg(kernel, 3, sizeof(cl_uint),(void*)&sortOrder);
#ifdef USE_SHARED_MEM
        clSetKernelArg(kernel, 4, (GROUP_SIZE << 1) *sizeof(cl_uint),NULL);
#elif def USE_SHARED_MEM_2
        clSetKernelArg(kernel, 5, (GROUP_SIZE << 2) *sizeof(cl_uint),NULL);
#endif 
        size_t globalThreads[1] = {LENGTH/2};
        size_t threadsPerGroup[1] = {GROUP_SIZE};

        for(cl_uint stage = 0; stage < stages; ++stage) {
            clSetKernelArg(kernel, 1, sizeof(cl_uint),(void*)&stage);

            for(cl_uint subStage = 0; subStage < stage +1; subStage++) {
                clSetKernelArg(kernel, 2, sizeof(cl_uint),(void*)&subStage);
				cl_event exeEvt; 
		        cl_ulong executionStart, executionEnd;
				error = clEnqueueNDRangeKernel(queue,
				                               kernel,
				                               1,
				                               NULL,
		                                       globalThreads,
		                                       threadsPerGroup, 
		                                       0,
		                                       NULL,
		                                       &exeEvt);
				clWaitForEvents(1, &exeEvt);
				if(error != CL_SUCCESS) {
					printf("Kernel execution failure!\n");
					exit(-22);
				}	
		
		        // let's understand how long it took?
		        clGetEventProfilingInfo(exeEvt, CL_PROFILING_COMMAND_START, sizeof(executionStart), &executionStart, NULL);
		        clGetEventProfilingInfo(exeEvt, CL_PROFILING_COMMAND_END, sizeof(executionEnd), &executionEnd, NULL);
		        clReleaseEvent(exeEvt);
		
		        printf("Execution of the bitonic sort took %lu.%lu s\n", (executionEnd - executionStart)/1000000000, (executionEnd - executionStart)%1000000000);
            }
        } 
        clEnqueueReadBuffer(queue,
                            device_A_in,
                            CL_TRUE,
                            0,
                            LENGTH * sizeof(cl_int),
                            host_A_out,
                            0,
                            NULL,
                            NULL);

        /* Clean up */
        for(i=0; i< NUMBER_OF_FILES; i++) { free(buffer[i]); }
        clReleaseProgram(program);
        clReleaseContext(context);
        clReleaseMemObject(device_A_in);
    }
    
    free(host_A_in);
    free(host_A_out);
}

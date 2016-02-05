#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <alloca.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#ifdef  __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "matrixmultiplication_config.h"

#define GROUP_SIZE 73 // ATI HD6870x2 has 14 parallel compute units
#define WIDTH_G 1024
#define HEIGHT_G 1024

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
compare(cl_int* gpuMatC, cl_int* matA, cl_int* matB, int heightA, int widthA, int widthB) {

    cl_int* cpuMat = (cl_int*) malloc(widthB * heightA * sizeof(cl_int));
    memset(cpuMat,0, widthB * heightA * sizeof(cl_int));
    for(int i = 0; i < heightA; ++i) 
        for(int j = 0; j < widthB; ++j) 
            for(int k = 0; k < widthA; ++k) 
                cpuMat[i * widthB + j] += (matA[i * widthA + k] * matB[k * widthB + j]);

    size_t length = heightA * widthB;
    for(int i =0 ; i < length; ++i) {
        //printf("cpu[%d] vs gpu[%d]\n", cpuMat[i], gpuMatC[i]);
        if (cpuMat[i] != gpuMatC[i]) return 0;
    }
    free(cpuMat);
    return 1;
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

void fillRandom(int* data, unsigned int width, unsigned height, unsigned int seed) {
    int* iptr = data;
    
    if(!seed) seed = (unsigned int) time(NULL);
    srand(seed);
    for(int i = 0 ; i < width; ++i) 
        for(int j = 0; j < height; ++j)
            iptr[j+i*width] = rand() % 100;
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

    cl_mem matrixAMemObj; // input matrix A mem buffer
    cl_mem matrixBMemObj; // input matrix B mem buffer
    cl_mem matrixCMemObj; // input matrix C mem buffer
    cl_int* matrixA;      // input matrix A
    cl_int* matrixB;      // input matrix B
    cl_int* matrixC;      // input matrix C
    cl_uint widthA = WIDTH_G;
    cl_uint heightA = HEIGHT_G;
    cl_uint widthB = WIDTH_G;
    cl_uint heightB = HEIGHT_G;

	{
	    // allocate memory for input and output matrices 
        // based on whatever matrix theory i know.
	    matrixA = (cl_int*)malloc(widthA * heightA * sizeof(cl_int));
	    matrixB = (cl_int*)malloc(widthB * heightB * sizeof(cl_int));
	    matrixC = (cl_int*)malloc(widthB * heightA * sizeof(cl_int));

	    memset(matrixA, 0, widthA * heightA * sizeof(cl_int));
	    memset(matrixB, 0, widthB * heightB * sizeof(cl_int));
	    memset(matrixC, 0, widthB * heightA * sizeof(cl_int));
        
        fillRandom(matrixA, widthA, heightA, 643);
        fillRandom(matrixB, widthB, heightB, 991);
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
        const char *file_names[] = {"simple_mm_mult.cl"};
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

        cl_kernel kernel = clCreateKernel(program, "mmmult", &error);

        matrixAMemObj = clCreateBuffer(context,
                                       CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                       widthA * heightA * sizeof(cl_int),
                                       matrixA,
                                       &error);

        matrixBMemObj = clCreateBuffer(context,
                                       CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                       widthB * heightB * sizeof(cl_int),
                                       matrixB,
                                       &error);

        matrixCMemObj = clCreateBuffer(context,
                                       CL_MEM_WRITE_ONLY|CL_MEM_ALLOC_HOST_PTR,
                                       widthB * heightA * sizeof(cl_int),
                                       0,
                                       &error);

        clSetKernelArg(kernel, 0, sizeof(cl_int),(void*)&widthB);
        clSetKernelArg(kernel, 1, sizeof(cl_int),(void*)&heightA);
        clSetKernelArg(kernel, 2, sizeof(cl_mem),(void*)&matrixAMemObj);
        clSetKernelArg(kernel, 3, sizeof(cl_mem),(void*)&matrixBMemObj);
        clSetKernelArg(kernel, 4, sizeof(cl_mem),(void*)&matrixCMemObj);
         
        size_t globalThreads[] = {widthB, heightA};

		cl_event exeEvt; 
        cl_ulong executionStart, executionEnd;
		error = clEnqueueNDRangeKernel(queue,
		                               kernel,
		                               2,
		                               NULL,
                                       globalThreads,
                                       NULL, 
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

        printf("Execution the matrix-matrix multiplication took %lu.%lu s\n", (executionEnd - executionStart)/1000000000, (executionEnd - executionStart)%1000000000);
 
        clEnqueueReadBuffer(queue,
                            matrixCMemObj,
                            CL_TRUE,
                            0,
                            widthB * heightA * sizeof(cl_int),
                            matrixC,
                            0,
                            NULL,
                            NULL);
       
        if (compare(matrixC, matrixA, matrixB, heightA, widthA, widthB))
            printf("Passed!\n");
        else 
            printf("Failed!\n");
 
        /* Clean up */
        for(i=0; i< NUMBER_OF_FILES; i++) { free(buffer[i]); }
        clReleaseProgram(program);
        clReleaseContext(context);
        clReleaseMemObject(matrixAMemObj);
        clReleaseMemObject(matrixBMemObj);
        clReleaseMemObject(matrixCMemObj);
    }
    
    free(matrixA);
    free(matrixB);
    free(matrixC);
}

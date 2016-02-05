#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <alloca.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef  __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define BIN_SIZE 256
#define GROUP_SIZE 128

void
calculateHostBin(int width, int height, cl_uint* hostBin, cl_uint* data) {
    for(int i = 0; i < width*height; i++ ) {
            hostBin[data[i]]++;
    }
#ifdef DEBUG
    for(int i = 0; i < BIN_SIZE; i++ ) {
            printf("binDataOnHost[%d]=%d ", i, hostBin[i]);
            if (i % 10 == 0) printf("\n");
    }
#endif
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
int main(int argc, char** argv) {
    int height; int width;       // dimensions of the data grid
    
    height = atoi(argv[1]);
    width  = atoi(argv[2]);
    
    /* OpenCL 1.1 data structures */
    cl_platform_id* platforms;
    cl_program program;
    cl_device_id device;
    cl_context context;
    cl_int subHistogramCount;
    cl_uint* data = NULL;
    cl_uint* hostBin = NULL;
    cl_int* deviceBin = NULL;
    cl_int* intermediateBins = NULL;
    cl_command_queue queue;
    cl_mem inputDataBuffer;
    cl_mem intermediateBinBuffer; 

    /* OpenCL 1.1 scalar data types */
    cl_int numOfPlatforms;
    cl_int  error;
    size_t globalThreads;
    size_t localThreads;
    
    /* Perform initialization of data structures & data */
    {
        int w = width;
        int h = height;
        
        // ensure a minimum size of 32-KB is present to work this.
        w = (w / BIN_SIZE ? w / BIN_SIZE: 1) * BIN_SIZE;
        h = (h / GROUP_SIZE ? h / GROUP_SIZE : 1) * GROUP_SIZE;
        
        subHistogramCount = (w * h) / (GROUP_SIZE * BIN_SIZE);
        data = (cl_uint*) malloc(w * h * sizeof(cl_uint));
        
        for(int i = 0; i < w * h; i++) data[i] = rand() % BIN_SIZE;
        
        hostBin = (cl_uint*) malloc(BIN_SIZE * sizeof(cl_uint));
        memset(hostBin, 0, BIN_SIZE * sizeof(cl_uint));
        
        intermediateBins = (cl_int*) malloc(BIN_SIZE * (subHistogramCount) * sizeof(cl_int));
        memset(intermediateBins, 0, BIN_SIZE * (subHistogramCount) * sizeof(cl_int));
        
        deviceBin = (cl_int*) malloc(BIN_SIZE * sizeof(cl_int));
        memset(deviceBin, 0, BIN_SIZE * sizeof(cl_int));

        globalThreads = (w * h)/ BIN_SIZE;
        localThreads  = GROUP_SIZE;
        width = w;
        height = h;
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
    // Search for a CPU/GPU device through the installed platforms
    // Build a OpenCL program and do not run it.
    for(cl_int i = 0; i < numOfPlatforms; i++ ) {
        // Get the GPU device
        error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);

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
        const char *file_names[] = {"histogram.cl"};
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
        
        printf("width=%d, height=%d, queue...global=%zd local=%zd subhistograms=%d.\n", width,height, globalThreads,
                                                                    localThreads,
                                                                    subHistogramCount);
        
        queue = clCreateCommandQueue(context, device, 0, &error);

        cl_kernel kernel = clCreateKernel(program, "histogram256", &error);

        inputDataBuffer = clCreateBuffer(context,
                                 CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                 width * height * sizeof(cl_uint),
                                 data,
                                 &error);

        intermediateBinBuffer = clCreateBuffer(context,
                                         CL_MEM_WRITE_ONLY,
                                         BIN_SIZE * subHistogramCount * sizeof(cl_uint),
                                         NULL,
                                         &error);

        clSetKernelArg(kernel, 0, sizeof(cl_mem),(void*)&inputDataBuffer);
        // the importance of uchar being that its unsigned char i.e. value range[0x00..0xff]
        clSetKernelArg(kernel, 1, BIN_SIZE * GROUP_SIZE * sizeof(cl_uchar), NULL); // bounded by LOCAL MEM SIZE in GPU
        clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&intermediateBinBuffer);
     
		cl_event exeEvt; 
		error = clEnqueueNDRangeKernel(queue,
		                               kernel,
		                               1,
		                               NULL, &globalThreads, &localThreads, 0, NULL, &exeEvt);
		clWaitForEvents(1, &exeEvt);
		if(error != CL_SUCCESS) {
			printf("Kernel execution failure!\n");
			exit(-22);
		}	
        clReleaseEvent(exeEvt);
 
        clEnqueueReadBuffer(queue,
                            intermediateBinBuffer,
                            CL_TRUE,
                            0,
                            subHistogramCount * BIN_SIZE * sizeof(cl_uint),
                            intermediateBins,
                            0,
                            NULL,
                            NULL);

        memset(deviceBin,0, BIN_SIZE * sizeof(cl_int));
       
        for(int i = 0; i < subHistogramCount; ++i)
            for( int j = 0; j < BIN_SIZE; ++j) {
		        printf("see(%d,%d)..%zd ",i,j, intermediateBins[i*BIN_SIZE+j]);
                if (j % 10 == 0) printf("\n");
                deviceBin[j] += intermediateBins[i * BIN_SIZE + j];
            }

        /* verify results */
        calculateHostBin(width, height, hostBin, data);
        
        int result = 1;
        for( int i = 0; i < BIN_SIZE; ++i)
            printf("comparing %d %d == %d?\n",i, hostBin[i], deviceBin[i]);
            if(hostBin[i] != deviceBin[i]) {
                result = 0;
                break;
            }
        if(result) {
            fprintf(stdout, "Passed!\n");
        } else {
            fprintf(stdout, "Failed\n");
        }
        
        
        /* Clean up */
        //for(cl_int i = 0; i < numOfKernels; i++) { clReleaseKernel(kernels[i]); }
        for(i=0; i< NUMBER_OF_FILES; i++) { free(buffer[i]); }
        clReleaseProgram(program);
        clReleaseContext(context);
        clReleaseMemObject(inputDataBuffer);
        clReleaseMemObject(intermediateBinBuffer);
    }
    
    free(data);
    free(hostBin);
    free(intermediateBins);
    free(deviceBin);

}

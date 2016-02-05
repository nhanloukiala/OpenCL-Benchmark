#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <alloca.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include "radixsort_config.h"
#include "common.h"

#ifdef  __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define GROUP_SIZE 64                   // ATI HD7870 has 20 parallel compute units, !!!wavefront programming!!!
#define BIN_SIZE 256

#define DATA_SIZE (1<<16)

cl_kernel histogramKernel;
cl_kernel permuteKernel  ;
cl_kernel unifiedBlockScanKernel ;
cl_kernel blockScanKernel ;
cl_kernel prefixSumKernel;
cl_kernel blockAddKernel ;
cl_kernel mergePrefixSumsKernel   ;
cl_mem unsortedData_d    ;
cl_mem histogram_d       ;
cl_mem scannedHistogram_d;
cl_mem sortedData_d      ;
cl_mem sum_in_d          ;
cl_mem sum_out_d         ;
cl_mem summary_in_d      ;
cl_mem summary_out_d     ;
cl_command_queue commandQueue;


int
waitAndReleaseDevice(cl_event* event) {
    cl_int status = CL_SUCCESS;
    cl_int eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE) {
        clGetEventInfo(*event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &eventStatus, NULL);
    }
    status = clReleaseEvent(*event);
    CHECK_ERROR(status, CL_SUCCESS, "Failed to release event\n");
    return 0;
}


// Translated from the COUNTING-SORT algorithm presented in their
// paper "Radix Sort for Vector Multiprocessors" by Zagha and Blelloch
int radixSortCPU(cl_uint* unsortedData, cl_uint* hSortedData) {

    cl_uint *histogram = (cl_uint*) malloc(R * sizeof(cl_uint));
    cl_uint *scratch = (cl_uint*) malloc(DATA_SIZE * sizeof(cl_uint));

    if(histogram != NULL && scratch != NULL) {

        memcpy(scratch, unsortedData, DATA_SIZE * sizeof(cl_uint));
        for(int bits = 0; bits < sizeof(cl_uint) * bitsbyte ; bits += bitsbyte) {

            // Initialize histogram bucket to zeros
            memset(histogram, 0, R * sizeof(cl_uint));

            // Calculate 256 histogram for all element
            for(int i = 0; i < DATA_SIZE; ++i)
            {
                cl_uint element = scratch[i];
                cl_uint value = (element >> bits) & R_MASK;
                histogram[value]++;
            }

            // Apply the prefix-sum algorithm to the histogram
            // the problem here is that if there the distribution of
            // values among the R-bins might not be "even", resulting
            // in more than 1 bin have zero values.
            // This is evident when applied to english words since certain
            // characters in the ASCII-based alphabet has a higher occurence
            // than other characters.
            cl_uint sum = 0;
            for(int i = 0; i < R; ++i)
            {
                cl_uint val = histogram[i];
                histogram[i] = sum;
                sum += val;
            }

            // Rearrange  the elements based on prescanned histogram
            // Thus far, the preceding code is basically adopted from
            // the "counting sort" algorithm.
            for(int i = 0; i < DATA_SIZE; ++i)
            {
                cl_uint element = scratch[i];
                cl_uint value = (element >> bits) & R_MASK;
                cl_uint index = histogram[value];
                hSortedData[index] = scratch[i];
                histogram[value] = index + 1;
            }

            // Copy to 'scratch' for further use
            if(bits != bitsbyte * 3)
                memcpy(scratch, hSortedData, DATA_SIZE * sizeof(cl_uint));
        }
    }

    free(scratch);
    free(histogram);
    return 1;
}

// This is the threaded-historgram which builds histograms
// and bins them based on a size of 256 or 1<<8
void computeHistogram(int currByte) {
    cl_event execEvt;
    cl_int status;
    size_t globalThreads = DATA_SIZE;
    size_t localThreads  = BIN_SIZE;
    status = clSetKernelArg(histogramKernel, 0, sizeof(cl_mem), (void*)&unsortedData_d);
    status = clSetKernelArg(histogramKernel, 1, sizeof(cl_mem), (void*)&histogram_d);
    status = clSetKernelArg(histogramKernel, 2, sizeof(cl_int), (void*)&currByte);
    status = clSetKernelArg(histogramKernel, 3, sizeof(cl_int) * BIN_SIZE, NULL); 
    status = clEnqueueNDRangeKernel(
        commandQueue, 
        histogramKernel,
        1,
        NULL,
        &globalThreads,
        &localThreads,
        0,
        NULL,
        &execEvt);
    clFlush(commandQueue);
    waitAndReleaseDevice(&execEvt);
}

void computeRankingNPermutations(int currByte, size_t groupSize) {
    cl_int status;
    cl_event execEvt;

    size_t globalThreads = DATA_SIZE/R;
    size_t localThreads  = groupSize;

    status = clSetKernelArg(permuteKernel, 0, sizeof(cl_mem), (void*)&unsortedData_d);
    status = clSetKernelArg(permuteKernel, 1, sizeof(cl_mem), (void*)&scannedHistogram_d);
    status = clSetKernelArg(permuteKernel, 2, sizeof(cl_int), (void*)&currByte);
    status = clSetKernelArg(permuteKernel, 3, groupSize * R * sizeof(cl_ushort), NULL); // shared memory
    status = clSetKernelArg(permuteKernel, 4, sizeof(cl_mem), (void*)&sortedData_d);

    status = clEnqueueNDRangeKernel(commandQueue, permuteKernel, 1, NULL, &globalThreads, &localThreads, 0, NULL, &execEvt);
    clFlush(commandQueue);
    waitAndReleaseDevice(&execEvt);

    cl_event copyEvt;
    status = clEnqueueCopyBuffer(commandQueue, sortedData_d, unsortedData_d, 0, 0, DATA_SIZE * sizeof(cl_uint), 0, NULL, &copyEvt);
    clFlush(commandQueue);
    waitAndReleaseDevice(&copyEvt);
}

void computeBlockScans() {
    cl_int status;
    
    size_t numOfGroups = DATA_SIZE / BIN_SIZE;
    size_t globalThreads[2] = {numOfGroups, R};
    size_t localThreads[2]  = {GROUP_SIZE, 1};
    cl_uint groupSize = GROUP_SIZE;

    status = clSetKernelArg(blockScanKernel, 0, sizeof(cl_mem), (void*)&scannedHistogram_d);
    status = clSetKernelArg(blockScanKernel, 1, sizeof(cl_mem), (void*)&histogram_d);
    status = clSetKernelArg(blockScanKernel, 2, GROUP_SIZE * sizeof(cl_uint), NULL);
    status = clSetKernelArg(blockScanKernel, 3, sizeof(cl_uint), &groupSize); 
    status = clSetKernelArg(blockScanKernel, 4, sizeof(cl_mem), &sum_in_d);

    cl_event execEvt;
    status = clEnqueueNDRangeKernel(
                commandQueue,
                blockScanKernel,
                2,
                NULL,
                globalThreads,
                localThreads,
                0, 
                NULL,
                &execEvt);
    clFlush(commandQueue);
    waitAndReleaseDevice(&execEvt);

    // If there is only 1 workgroup, we will skip the block-addition and prefix-sum kernel
    if(numOfGroups/GROUP_SIZE != 1) {
        size_t globalThreadsPrefix[2] = {numOfGroups/GROUP_SIZE, R};
        status = clSetKernelArg(prefixSumKernel, 0, sizeof(cl_mem), (void*)&sum_out_d);
        status = clSetKernelArg(prefixSumKernel, 1, sizeof(cl_mem), (void*)&sum_in_d);
        status = clSetKernelArg(prefixSumKernel, 2, sizeof(cl_mem), (void*)&summary_in_d);
        cl_uint stride = (cl_uint)numOfGroups/GROUP_SIZE;
        status = clSetKernelArg(prefixSumKernel, 3, sizeof(cl_uint), (void*)&stride);
        cl_event prefixSumEvt;
        status = clEnqueueNDRangeKernel(
                    commandQueue,
                    prefixSumKernel,
                    2,
                    NULL,
                    globalThreadsPrefix,
                    NULL,
                    0,
                    NULL,
                    &prefixSumEvt);
        clFlush(commandQueue);
        waitAndReleaseDevice(&prefixSumEvt);
        
        // Run block-addition kernel
        cl_event execEvt2;
        size_t globalThreadsAdd[2] = {numOfGroups, R};
        size_t localThreadsAdd[2]  = {GROUP_SIZE, 1};
        status = clSetKernelArg(blockAddKernel, 0, sizeof(cl_mem), (void*)&sum_out_d);  
        status = clSetKernelArg(blockAddKernel, 1, sizeof(cl_mem), (void*)&scannedHistogram_d);  
        status = clSetKernelArg(blockAddKernel, 2, sizeof(cl_uint), (void*)&stride);  
        status = clEnqueueNDRangeKernel(
                    commandQueue,
                    blockAddKernel,
                    2,
                    NULL,
                    globalThreadsAdd,
                    localThreadsAdd,
                    0,
                    NULL,
                    &execEvt2);
        clFlush(commandQueue);
        waitAndReleaseDevice(&execEvt2);

        // Run parallel array scan since we have GROUP_SIZE values which are summarized from each row
        // and we accumulate them
        size_t globalThreadsScan[1] = {R};
        size_t localThreadsScan[1] = {R};
        status = clSetKernelArg(unifiedBlockScanKernel, 0, sizeof(cl_mem), (void*)&summary_out_d);
        if(numOfGroups/GROUP_SIZE != 1) 
            status = clSetKernelArg(unifiedBlockScanKernel, 1, sizeof(cl_mem), (void*)&summary_in_d); 
        else
            status = clSetKernelArg(unifiedBlockScanKernel, 1, sizeof(cl_mem), (void*)&sum_in_d); 

        status = clSetKernelArg(unifiedBlockScanKernel, 2, R * sizeof(cl_uint), NULL);  // shared memory
        groupSize = R;
        status = clSetKernelArg(unifiedBlockScanKernel, 3, sizeof(cl_uint), (void*)&groupSize); 
        cl_event execEvt3;
        status = clEnqueueNDRangeKernel(
                    commandQueue,
                    unifiedBlockScanKernel,
                    1,
                    NULL,
                    globalThreadsScan,
                    localThreadsScan,
                    0, 
                    NULL, 
                    &execEvt3);
        clFlush(commandQueue);
        waitAndReleaseDevice(&execEvt3);

        cl_event execEvt4;
        size_t globalThreadsOffset[2] = {numOfGroups, R};
        status = clSetKernelArg(mergePrefixSumsKernel, 0, sizeof(cl_mem), (void*)&summary_out_d);
        status = clSetKernelArg(mergePrefixSumsKernel, 1, sizeof(cl_mem), (void*)&scannedHistogram_d);
        status = clEnqueueNDRangeKernel(commandQueue, mergePrefixSumsKernel, 2, NULL, globalThreadsOffset, NULL, 0, NULL, &execEvt4);
        clFlush(commandQueue);
        waitAndReleaseDevice(&execEvt4);
    }
}

// Function that invokes the execution of the kernels
void runKernels(cl_uint* dSortedData,
                size_t numOfGroups,
                size_t groupSize) {

    for(int currByte = 0; currByte < sizeof(cl_uint) * bitsbyte ; currByte += bitsbyte) {
        computeHistogram(currByte);
        computeBlockScans();
        computeRankingNPermutations(currByte, groupSize);
    }

    cl_int status;
    cl_uint* data = (cl_uint*)clEnqueueMapBuffer(commandQueue, sortedData_d, CL_TRUE, CL_MAP_READ, 0, DATA_SIZE*sizeof(cl_uint),0,NULL,NULL,&status);
    CHECK_ERROR(status, CL_SUCCESS, "mapping buffer failed");
    memcpy(dSortedData, data, DATA_SIZE*sizeof(cl_uint));
    clEnqueueUnmapMemObject(commandQueue,sortedData_d,data,0,NULL,NULL);
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

void fillRandom(cl_uint* data, unsigned int length) {
    cl_uint* iptr = data;
    
    for(int i = 0 ; i < length; ++i) 
            iptr[i] = (cl_uint)rand();
}

int main(int argc, char** argv) {
    cl_uint* unsortedData = NULL;
    cl_uint* dSortedData = NULL;
    cl_uint* hSortedData = NULL;

    /* OpenCL 1.1 data structures */
    cl_platform_id* platforms;
    cl_device_id device;
    cl_uint numOfPlatforms;
    cl_int  error;
    cl_program program;
    cl_uint groupSize = GROUP_SIZE;
    cl_uint numOfGroups = DATA_SIZE / (groupSize * R); 

    unsortedData = (cl_uint*) malloc(DATA_SIZE * sizeof(cl_uint));
    fillRandom(unsortedData, DATA_SIZE);

    dSortedData = (cl_uint*) malloc(DATA_SIZE * sizeof(cl_uint));
	memset(dSortedData, 0, DATA_SIZE * sizeof(cl_uint));
    
    hSortedData = (cl_uint*) malloc(DATA_SIZE * sizeof(cl_uint)); 
	memset(hSortedData, 0, DATA_SIZE * sizeof(cl_uint));

    //Get the number of platforms
    //Remember that for each vendor's SDK installed on the computer,
    //the number of available platform also increased.
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
        // Create a context 
        cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
        if(error != CL_SUCCESS) {
            perror("Can't create a valid OpenCL context");
            exit(1);
        }
        
        // Load the two source files into temporary datastores
        const char *file_names[] = {"RadixSort.cl"};
        const int NUMBER_OF_FILES = 1;
        char* buffer[NUMBER_OF_FILES];
        size_t sizes[NUMBER_OF_FILES];
        loadProgramSource(file_names, NUMBER_OF_FILES, buffer, sizes);
        
        // Create the OpenCL program object 
        program = clCreateProgramWithSource(context, NUMBER_OF_FILES, (const char**)buffer, sizes, &error);
	    if(error != CL_SUCCESS) {
            perror("Can't create the OpenCL program object");
            exit(1);
	    }
        // Build OpenCL program object and dump the error message, if any 
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

        commandQueue = clCreateCommandQueue(context, device, props, &error);
        CHECK_ERROR(error, CL_SUCCESS, "failed to create command queue");

        unsortedData_d     = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, sizeof(cl_uint) * DATA_SIZE, unsortedData, &error);
        CHECK_ERROR(error, CL_SUCCESS, "failed to allocate unsortedData_d");
        histogram_d        = clCreateBuffer(context, CL_MEM_READ_WRITE, numOfGroups * groupSize * R * sizeof(cl_uint), NULL, &error);
        CHECK_ERROR(error, CL_SUCCESS, "failed to allocate histogram_d");
        scannedHistogram_d = clCreateBuffer(context, CL_MEM_READ_WRITE, numOfGroups * groupSize * R * sizeof(cl_uint), NULL, &error);
        CHECK_ERROR(error, CL_SUCCESS, "failed to allocate scannedHistogram_d");
        sortedData_d       = clCreateBuffer(context, CL_MEM_WRITE_ONLY, DATA_SIZE * sizeof(cl_uint), NULL, &error);
        CHECK_ERROR(error, CL_SUCCESS, "failed to allocate sortedData_d");
        sum_in_d           = clCreateBuffer(context, CL_MEM_READ_WRITE, (DATA_SIZE/groupSize) * sizeof(cl_uint), NULL, &error);
        CHECK_ERROR(error, CL_SUCCESS, "failed to allocate sum_in_d");
        sum_out_d          = clCreateBuffer(context, CL_MEM_READ_WRITE, (DATA_SIZE/groupSize) * sizeof(cl_uint), NULL, &error);
        CHECK_ERROR(error, CL_SUCCESS, "failed to allocate sum_out_d");
        summary_in_d       = clCreateBuffer(context, CL_MEM_READ_WRITE, R * sizeof(cl_uint), NULL, &error);
        CHECK_ERROR(error, CL_SUCCESS, "failed to allocate summary_in_d");
        summary_out_d      = clCreateBuffer(context, CL_MEM_READ_WRITE, R * sizeof(cl_uint), NULL, &error);
        CHECK_ERROR(error, CL_SUCCESS, "failed to allocate summary_out_d");

        histogramKernel = clCreateKernel(program, "computeHistogram", &error);
        CHECK_ERROR(error, CL_SUCCESS, "Failed to create histogram kernel");
        permuteKernel   = clCreateKernel(program, "rankNPermute", &error);
        CHECK_ERROR(error, CL_SUCCESS, "Failed to create permute kernel");
        unifiedBlockScanKernel  = clCreateKernel(program, "unifiedBlockScan", &error);
        CHECK_ERROR(error, CL_SUCCESS, "Failed to create unifiedBlockScan kernel");
        blockScanKernel  = clCreateKernel(program, "blockScan", &error);
        CHECK_ERROR(error, CL_SUCCESS, "Failed to create blockScan kernel");
        prefixSumKernel = clCreateKernel(program, "blockPrefixSum", &error);
        CHECK_ERROR(error, CL_SUCCESS, "Failed to create compute block prefix sum kernel");
        blockAddKernel  = clCreateKernel(program, "blockAdd", &error);
        CHECK_ERROR(error, CL_SUCCESS, "Failed to create block addition kernel");
        mergePrefixSumsKernel    = clCreateKernel(program, "mergePrefixSums", &error);
        CHECK_ERROR(error, CL_SUCCESS, "Failed to create fix offset kernel");

printf("elementCount: %u, numOfGroups: %u, groupSize: %u\n", DATA_SIZE, numOfGroups, groupSize);
        runKernels(
                   dSortedData,
                   numOfGroups, groupSize);


        // Clean up 
        for(int j=0; j< NUMBER_OF_FILES; j++) { free(buffer[j]); }

        clReleaseMemObject(unsortedData_d);
        clReleaseMemObject(histogram_d);
        clReleaseMemObject(scannedHistogram_d);
        clReleaseMemObject(sortedData_d);
        clReleaseMemObject(sum_in_d);
        clReleaseMemObject(sum_out_d);
        clReleaseMemObject(summary_out_d);
        clReleaseMemObject(summary_out_d);
        clReleaseKernel(histogramKernel);
        clReleaseKernel(permuteKernel);
        clReleaseKernel(unifiedBlockScanKernel);
        clReleaseKernel(blockScanKernel);
        clReleaseKernel(prefixSumKernel);
        clReleaseKernel(blockAddKernel);
        clReleaseKernel(mergePrefixSumsKernel);

        // Verification Checks
        radixSortCPU(unsortedData, hSortedData);
        for(int k = 0, acc = 0; k < DATA_SIZE; k++) { 
            if (hSortedData[k] == dSortedData[k]) acc++;
            if ((k+1) == DATA_SIZE) {
                if (acc == DATA_SIZE) printf("Passed:%u!\n", acc); else printf("Failed:%u!\n", acc);
            }
        }
        
    free(unsortedData);
    free(dSortedData);
    free(hSortedData);
    }
    
}

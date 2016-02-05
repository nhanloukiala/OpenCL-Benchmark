#include <stdio.h>
#include <stdlib.h>
 
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
void CL_CALLBACK postProcess(cl_event event, cl_int status, void *data) { 
  printf("%s\n", (char*)data);
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


int main()
{
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem objA = NULL;
    cl_mem objB = NULL;
    cl_mem objC = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;
 
    cl_event event1;
 
    int i, j;
    float *A;
    float *B;
    float *C;
 
    A = (float *)malloc(4*4*sizeof(float));
    B = (float *)malloc(4*4*sizeof(float));
    C = (float *)malloc(4*4*sizeof(float));
 
    /* Initialize input data */
    for (i=0; i<4; i++) {
        for (j=0; j<4; j++) {
            A[i*4+j] = i*4+j+1;
            B[i*4+j] = j*4+i+1;
        }
    }
 
    /* Get Platform/Device Information*/
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
 
    /* Create OpenCL Context */
    context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
 
    /* Create command queue */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
 
    /* Create Buffer Object */
    objA = clCreateBuffer(context, CL_MEM_READ_WRITE, 4*4*sizeof(float), NULL, &ret);
    objB = clCreateBuffer(context, CL_MEM_READ_WRITE, 4*4*sizeof(float), NULL, &ret);
    objC = clCreateBuffer(context, CL_MEM_READ_WRITE, 4*4*sizeof(float), NULL, &ret);
 
    /*
     * Creating an user event
     * As a user event is created, its execution status is set to be CL_SUBMITTED
     * and we tag the event to a callback so when event reaches CL_COMPLETE, it will 
     * execute postProcess
     */ 
    event1 = clCreateUserEvent(context, &ret);
    clSetEventCallback(event1, CL_COMPLETE, &postProcess, "Looks like its done.");

    /* Copy input data to the memory buffer */
 
    ret = clEnqueueWriteBuffer(command_queue, objA, CL_TRUE, 0, 4*4*sizeof(float), A, 0, NULL, NULL );
    printf("A has been written\n");
 
    /* The next command will wait for event1 according to its status*/
    ret = clEnqueueWriteBuffer(command_queue, objB, CL_TRUE, 0, 4*4*sizeof(float), B, 1, &event1, NULL);
    printf("B has been written\n");

    /* Tell event1 to complete */
    clSetUserEventStatus(event1, CL_COMPLETE);
	const char *file_names[] = {"sample_kernel.cl"}; 
	const int NUMBER_OF_FILES = 1;
	char* buffer[NUMBER_OF_FILES];
	size_t sizes[NUMBER_OF_FILES];
	loadProgramSource(file_names, NUMBER_OF_FILES, buffer, sizes);
	
    /* Create kernel program from source file*/
    program = clCreateProgramWithSource(context, 1, (const char **)buffer, sizes, &ret);
    ret     = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
 
    /* Create data parallel OpenCL kernel */
    kernel = clCreateKernel(program, "sample", &ret);
 
    /* Set OpenCL kernel arguments */
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&objA);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&objB);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&objC);
 
    size_t global_item_size = 4;
    size_t local_item_size = 1;
 
    /* Execute OpenCL kernel as data parallel */
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
                                 &global_item_size, &local_item_size, 0, NULL, NULL);
 
    /* Transfer result to host */
    ret = clEnqueueReadBuffer(command_queue, objC, CL_TRUE, 0, 4*4*sizeof(float), C, 0, NULL, NULL);
 
    /* Display Results */
    for (i=0; i<4; i++) {
        for (j=0; j<4; j++) {
            printf("%7.2f ", C[i*4+j]);
        }
        printf("\n");
    }
 
 
    /* Finalization */
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(objA);
    ret = clReleaseMemObject(objB);
    ret = clReleaseMemObject(objC);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
 
    free(A);
    free(B);
    free(C);
 
    return 0;
}

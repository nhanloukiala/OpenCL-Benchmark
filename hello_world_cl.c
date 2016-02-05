#include <stdio.h>
#include <math.h>
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.h>

const char *KernelSource = "\n" \
"__kernel void hello( \n" \
" __global char* a, \n" \
" __global char* b, \n" \
" __global char* c, \n" \
" const unsigned int count) \n" \
"{ \n" \
" int i = get_global_id(0); \n" \
" if(i < count) \n" \
" c[i] = a[i] + b[i]; \n" \
"} \n" \
"\n";

#define DATA_SIZE (16)

int main(int argc, char** argv)
{
	int err; // error code returned from api calls
	cl_device_id device_id; // compute device id
	cl_context context; // compute context
	cl_command_queue commands; // compute command queue
	cl_program program; // compute program
	cl_kernel kernel; // compute kernel
	cl_mem input; // device memory used for the input array
	cl_mem input2; // device memory used for the input array
	cl_mem output; // device memory used for the output array
	size_t global; // global domain size for our calculation
	size_t local; // local domain size for our calculation
	
	int i;
	unsigned int count = DATA_SIZE;

	char a[DATA_SIZE] = "Hello \0\0\0\0\0\0";
	char b[DATA_SIZE] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	char c[DATA_SIZE];

	printf("%s", a);

	cl_platform_id platform;
	unsigned int no_plat;

	// Get available platforms (this example uses the first available platform.
	// It may be a problem if the first platform does not include device we are looking for)
	err = clGetPlatformIDs(1,&platform,&no_plat);
	if (err != CL_SUCCESS) return -1;
	// Obtain the list of GPU devices available on the platform
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	if (err != CL_SUCCESS) return -1;
	// Create OpenCL context for the GPU device
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context) return -1;
	// This command is deprecated in OpenCL 2.0. 
	// clCreateCommandQueue() should be replaced with clCreateCommandQueueProperties(), and without declaring CL_USE_DEPRECATED_OPENCL_2_0_APIS a warning is raised
	commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands) return -1;
	// Create program object for the context that stores the source code specified by the KernelSource text string
	program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
	if (!program) return -1;
	// Compile and link the kernel program
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) return -1;
	// Create the OpenCL kernel  
	kernel = clCreateKernel(program, "hello", &err);
	if (!kernel || err != CL_SUCCESS) return -1;
	
	// Create memory objects and transfer the data "a" to the memory buffer
	input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, \
	sizeof(char) * DATA_SIZE, a, NULL);
	// Create memory objects and transfer the data "b" to the memory buffer
	input2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, \
	sizeof(char) * DATA_SIZE, b, NULL);
	// Create device memory buffer for the output data
	output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(char) * DATA_SIZE, NULL, NULL);
	if (!input || !output) return -1;
	
	// Set the kernel arguments
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &input2);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
	err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &count);
	if (err != CL_SUCCESS) return -1;

	local = DATA_SIZE;
	global = DATA_SIZE; // count;
	// Execute the OpenCL kernel in data parallel
	err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if (err != CL_SUCCESS) return -1;
	
	// Wait for the commands to get executed before reading back the results 
	clFinish(commands);
	// Read kernel results to the host memory buffer "c"
	err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(char) * count, c, \
	0, NULL, NULL );
	if (err != CL_SUCCESS) return -1;

	// The kernel input char arrays "Hello", and array "15,10,6,0,-11,1" and adds them together which produces the result array "World!"
	printf("%s\n", c);
	
	clReleaseMemObject(input);
	clReleaseMemObject(output);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
	return 0;
}
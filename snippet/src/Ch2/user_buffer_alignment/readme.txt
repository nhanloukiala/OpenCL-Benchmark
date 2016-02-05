================================
 AMD APP SDK v2.7 Profiler
================================
To get the profiling to work, you need to install the toolkit from AMD.

Download it from http://developer.amd.com/sdks/AMDAPPSDK/downloads/Pages/default.aspx

Take heed that you need GPU devices newer than "AMD Radeon HD 4000" series
and if you wish to profile DirectCompute kernels, then you need GPUs more recent
than "AMD Radeon HD 5000" series.

================================
Installing it
================================
The profiler comes as part of AMD APP SDK package. Once that's installed, 
you'll have the profiler.


==================================
HOWTO profile your OpenCL kernels
==================================
Navigate to where you installed the SDK, on Linux you can simply "cd" there
in a terminal.

> cd $AMDAPPSDKROOT

Next, locate the directory "tools" and the profiler is right "below" it.
The basic command to profile your kernels is

> <path to 'sprofile'>/sprofile [options] <InputApplication> <InputApplication's command line arguments>
e.g. In the sources provided, navigate to "Ch2/user_buffer_alignment" and run the command 

> /opt/AMDAPP/tools/AMDAPPProfiler-2.5/x86_64/sprofile -o perf.log -k all ./user_buffer_align

and you'll an output similar to the following :

		|= AMD APP Profiler V2.5.1804 is Enabled                   =|
		|= Number of OpenCL platforms found: 1                     =|
		|= Number of detected OpenCL devices: 2                    =|
		|= Kernel name: hello with arity: 1                        =|
		|= About to create command queue and enqueue this kernel...=|
		|= Writing to file : ./sp_tmp.hello__k1_Barts1.il          =|
		|= Writing to file : ./sp_tmp.hello__k1_Barts1.isa         =|
		|= Writing to file : ./sp_tmp.hello__k1_Barts1.cl          =|
		|= ........                                                =|

In this example, you'll see a file named "perf.log.csv" deposited locally and you can view it.


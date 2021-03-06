# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/local/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/tayboonl/PACKT_OpenCL_Book

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/tayboonl/PACKT_OpenCL_Book

# Include any dependencies generated for this target.
include src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/depend.make

# Include the progress variables for this target.
include src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/progress.make

# Include the compile flags for this target's objects.
include src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/flags.make

src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/RadixSort.c.o: src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/flags.make
src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/RadixSort.c.o: src/Ch10/RadixSort_GPU/RadixSort.c
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/tayboonl/PACKT_OpenCL_Book/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/RadixSort.c.o"
	cd /Users/tayboonl/PACKT_OpenCL_Book/src/Ch10/RadixSort_GPU && /usr/bin/gcc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/RadixSort_GPU.dir/RadixSort.c.o   -c /Users/tayboonl/PACKT_OpenCL_Book/src/Ch10/RadixSort_GPU/RadixSort.c

src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/RadixSort.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/RadixSort_GPU.dir/RadixSort.c.i"
	cd /Users/tayboonl/PACKT_OpenCL_Book/src/Ch10/RadixSort_GPU && /usr/bin/gcc  $(C_DEFINES) $(C_FLAGS) -E /Users/tayboonl/PACKT_OpenCL_Book/src/Ch10/RadixSort_GPU/RadixSort.c > CMakeFiles/RadixSort_GPU.dir/RadixSort.c.i

src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/RadixSort.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/RadixSort_GPU.dir/RadixSort.c.s"
	cd /Users/tayboonl/PACKT_OpenCL_Book/src/Ch10/RadixSort_GPU && /usr/bin/gcc  $(C_DEFINES) $(C_FLAGS) -S /Users/tayboonl/PACKT_OpenCL_Book/src/Ch10/RadixSort_GPU/RadixSort.c -o CMakeFiles/RadixSort_GPU.dir/RadixSort.c.s

src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/RadixSort.c.o.requires:
.PHONY : src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/RadixSort.c.o.requires

src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/RadixSort.c.o.provides: src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/RadixSort.c.o.requires
	$(MAKE) -f src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/build.make src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/RadixSort.c.o.provides.build
.PHONY : src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/RadixSort.c.o.provides

src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/RadixSort.c.o.provides.build: src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/RadixSort.c.o

# Object files for target RadixSort_GPU
RadixSort_GPU_OBJECTS = \
"CMakeFiles/RadixSort_GPU.dir/RadixSort.c.o"

# External object files for target RadixSort_GPU
RadixSort_GPU_EXTERNAL_OBJECTS =

src/Ch10/RadixSort_GPU/RadixSort_GPU: src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/RadixSort.c.o
src/Ch10/RadixSort_GPU/RadixSort_GPU: src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/build.make
src/Ch10/RadixSort_GPU/RadixSort_GPU: src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking C executable RadixSort_GPU"
	cd /Users/tayboonl/PACKT_OpenCL_Book/src/Ch10/RadixSort_GPU && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/RadixSort_GPU.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/build: src/Ch10/RadixSort_GPU/RadixSort_GPU
.PHONY : src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/build

src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/requires: src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/RadixSort.c.o.requires
.PHONY : src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/requires

src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/clean:
	cd /Users/tayboonl/PACKT_OpenCL_Book/src/Ch10/RadixSort_GPU && $(CMAKE_COMMAND) -P CMakeFiles/RadixSort_GPU.dir/cmake_clean.cmake
.PHONY : src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/clean

src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/depend:
	cd /Users/tayboonl/PACKT_OpenCL_Book && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/tayboonl/PACKT_OpenCL_Book /Users/tayboonl/PACKT_OpenCL_Book/src/Ch10/RadixSort_GPU /Users/tayboonl/PACKT_OpenCL_Book /Users/tayboonl/PACKT_OpenCL_Book/src/Ch10/RadixSort_GPU /Users/tayboonl/PACKT_OpenCL_Book/src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/Ch10/RadixSort_GPU/CMakeFiles/RadixSort_GPU.dir/depend


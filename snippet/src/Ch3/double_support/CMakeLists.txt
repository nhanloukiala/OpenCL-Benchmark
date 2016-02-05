cmake_minimum_required(VERSION 2.8)

option (DEBUG "Built binary in debug mode & 'printf' mostly" ON)
option (fp64 "cl_khr_fp64 support" ON)

find_package(OpenCL REQUIRED)

configure_file("./double_config.h.in" "./double_config.h")

if(CMAKE_COMPILER_IS_GNUCC)
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set (COMPILE_ARCH -m64)
    endif()
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86")
        set (COMPILE_ARCH -m32)
    endif()

    if (DEBUG)
        set (CMAKE_C_FLAGS "-std=c99 -Wall -DUNIX -g -DDEBUG ${COMPILE_ARCH}")
    else()
        set (CMAKE_C_FLAGS "-std=c99 -Wall -DUNIX ${COMPILE_ARCH}")
    endif()

    add_executable(DoubleSupport double_support.c)
    target_link_libraries(DoubleSupport ${OPENCL_LIBRARIES} )
    configure_file(double_support.cl ${CMAKE_CURRENT_BINARY_DIR}/double_support.cl COPYONLY)

endif(CMAKE_COMPILER_IS_GNUCC)


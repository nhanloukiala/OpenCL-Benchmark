cmake_minimum_required(VERSION 2.8)

option (DEBUG "debug build" ON)
option (DEBUG_VERBOSE "debug 'printf'" ON)

configure_file("./bitonicsort_config.h.in" "./bitonicsort_config.h")

if(CMAKE_COMPILER_IS_GNUCC)
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set (COMPILE_ARCH -m64)
    endif()
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86")
        set (COMPILE_ARCH -m32)
    endif()

    if (DEBUG)
        set (CMAKE_C_FLAGS "-std=c99 -Wall -DUNIX -g -DDEBUG ${COMPILE_ARCH} ${SSE_FLAGS}")
    else()
        set (CMAKE_C_FLAGS "-std=c99 -Wall -DUNIX ${COMPILE_ARCH} ${SSE_FLAGS}")
    endif()

    add_executable(BitonicSort_GPU_01 BitonicSort.c)
    target_link_libraries(BitonicSort_GPU_01 ${OPENCL_LIBRARIES} m)
    configure_file(BitonicSort.cl ${CMAKE_CURRENT_BINARY_DIR}/BitonicSort.cl COPYONLY)

endif(CMAKE_COMPILER_IS_GNUCC)


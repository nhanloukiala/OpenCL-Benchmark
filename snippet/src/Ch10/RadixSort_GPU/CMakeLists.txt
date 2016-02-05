cmake_minimum_required(VERSION 2.8)

option (DEBUG "debug build" ON)
option (DEBUG_VERBOSE "debug 'printf'" ON)

include_directories("../common")

configure_file("./radixsort_config.h.in" "./radixsort_config.h")

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

    add_executable(RadixSort_GPU RadixSort.c)
    target_link_libraries(RadixSort_GPU ${OPENCL_LIBRARIES} m)
    configure_file(RadixSort.cl ${CMAKE_CURRENT_BINARY_DIR}/RadixSort.cl COPYONLY)

endif(CMAKE_COMPILER_IS_GNUCC)


cmake_minimum_required(VERSION 2.8)

option (DEBUG "debug build and 'printf'" ON)

configure_file("./fma_mad_cmp_config.h.in" "./fma_mad_cmp_config.h")

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

    add_executable(FmaMadCmp fma_mad_cmp.c)
    target_link_libraries(FmaMadCmp ${OPENCL_LIBRARIES} )
    configure_file(fma_mad_cmp.cl ${CMAKE_CURRENT_BINARY_DIR}/fma_mad_cmp.cl COPYONLY)

endif(CMAKE_COMPILER_IS_GNUCC)


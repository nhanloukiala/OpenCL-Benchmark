cmake_minimum_required(VERSION 2.8)

option (DEBUG "debug build and 'printf'" ON)

find_package(SSE REQUIRED)

configure_file("./vectorization_config.h.in" "./vectorization_config.h")

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

    add_executable(Vectorization vectorization.c)
    target_link_libraries(Vectorization ${OPENCL_LIBRARIES} )
    configure_file(vectorization.cl ${CMAKE_CURRENT_BINARY_DIR}/vectorization.cl COPYONLY)

endif(CMAKE_COMPILER_IS_GNUCC)


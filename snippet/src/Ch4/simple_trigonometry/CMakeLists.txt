cmake_minimum_required(VERSION 2.8)

option (DEBUG "debug build and 'printf'" ON)

configure_file("./simple_trigo_config.h.in" "./simple_trigo_config.h")

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

    add_executable(SimpleTrigo simple_trigo.c)
    target_link_libraries(SimpleTrigo ${OPENCL_LIBRARIES} )
    configure_file(simple_trigo.cl ${CMAKE_CURRENT_BINARY_DIR}/simple_trigo.cl COPYONLY)

endif(CMAKE_COMPILER_IS_GNUCC)


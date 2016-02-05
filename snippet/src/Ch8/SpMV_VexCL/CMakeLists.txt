
find_path(BOOST_INCLUDE_DIRS boost PATHS /usr/local/include /usr/include)
find_library(BOOST_SYS_LIBRARIES NAMES boost_system PATHS /usr/local/lib /usr/lib)
find_library(BOOST_CHRONO_LIBRARIES NAMES boost_chrono PATHS /usr/local/lib /usr/lib)

include_directories(
    ${BOOST_INCLUDE_DIRS}
    ${VexCL_INCLUDE_DIR}
    )

set(CMAKE_CXX_FLAGS "-std=c++0x")

add_executable(SpMV_VexCL SpMV.cpp)
target_link_libraries(SpMV_VexCL ${OPENCL_LIBRARIES} ${BOOST_SYS_LIBRARIES} ${BOOST_CHRONO_LIBRARIES})
set_target_properties(SpMV_VexCL PROPERTIES COMPILE_FLAGS -Wno-comment)

cmake_minimum_required(VERSION 3.10)

# set(CC "D:/mingw64/bin/gcc.exe")
# set(CMAKE_C_COMPILER "D:/mingw64/bin/gcc.exe")

# set(CXX "D:/mingw64/bin/g++.exe")
# set(CMAKE_CXX_COMPILER "D:/mingw64/bin/g++.exe")
message(${CMAKE_BINARY_DIR})
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib/)
# set(CMAKE_CXX_FLAGS "/GR")


# set(CMAKE_DEBUG_POSTFIX ".pyd") 
# set(CMAKE_RELEASE_POSTFIX ".pyd") 

project(Tutorial)
include_directories(${CMAKE_BINARY_DIR}/../lib/)
include_directories(${CMAKE_BINARY_DIR}/../lib/xtensor/include/)
include_directories(${CMAKE_BINARY_DIR}/../lib/xtl/include/)
include_directories(${CMAKE_BINARY_DIR}/../lib/xtensor-blas/include/)
include_directories(${CMAKE_BINARY_DIR}/../lib/pybind11/include/)
# include_directories(${CMAKE_BINARY_DIR}/../src/pybind11/include)
# include_directories(D:/anaconda3/envs/pybind_demo/include)
# find_package(D:/anaconda3/envs/pybind_demo/Lib/site-packages/pybind11/share/cmake/pybind11 REQUIRED)

add_subdirectory(${CMAKE_BINARY_DIR}/../lib/pybind11)
add_subdirectory(${CMAKE_BINARY_DIR}/../pyd_make)


# add_library(sub SHARED main.cpp)
# add_executable(Tutorial main.cpp fun1.cpp fun1.h)


# pybind11_add_module(example src/main.cpp)

# target_link_directories(sub PUBLIC D:/anaconda3/envs/pybind_demo/libs)
# target_link_libraries(Tutorial PUBLIC sub)

cmake_minimum_required(VERSION 3.10)

# set(CC "D:/mingw64/bin/gcc.exe")
# set(CMAKE_C_COMPILER "D:/mingw64/bin/gcc.exe")

# set(CXX "D:/mingw64/bin/g++.exe")
# set(CMAKE_CXX_COMPILER "D:/mingw64/bin/g++.exe")
message(${CMAKE_BINARY_DIR})

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib/)

# set(CMAKE_DEBUG_POSTFIX ".pyd")
# set(CMAKE_RELEASE_POSTFIX ".pyd")
include_directories(${CMAKE_BINARY_DIR}/../lib/)
include_directories(${CMAKE_BINARY_DIR}/../lib/xtensor/include/)
include_directories(${CMAKE_BINARY_DIR}/../lib/xtl/include/)
include_directories(${CMAKE_BINARY_DIR}/../lib/pybind11/include/)
include_directories(${CMAKE_BINARY_DIR}/../lib/xtensor-python/include/)
include_directories(D:/anaconda3/Lib/site-packages/numpy/core/include)

# set(CMAKE_CXX_FLAGS "/GR")

# include_directories(${CMAKE_BINARY_DIR}/../src/pybind11/include)

# include_directories(D:/anaconda3/envs/pybind_demo/include)
# find_package(D:/anaconda3/envs/pybind_demo/Lib/site-packages/pybind11/share/cmake/pybind11 REQUIRED)

# add_subdirectory(${CMAKE_BINARY_DIR}/../debug/pybind11)
# add_subdirectory(${CMAKE_BINARY_DIR}/../src)

# add_library(sub SHARED main.cpp)
# add_executable(Tutorial main.cpp fun1.cpp fun1.h)

project(data_node_float32)
pybind11_add_module(data_node_float32 ${CMAKE_BINARY_DIR}/../pyd_make/data_node_f32.cpp)

project(opt_node_float32)
pybind11_add_module(opt_node_float32 ${CMAKE_BINARY_DIR}/../pyd_make/opt_node_f32.cpp)

project(cal_map_flow_float32)
pybind11_add_module(cal_map_flow_float32 ${CMAKE_BINARY_DIR}/../pyd_make/cal_map_flow_f32.cpp)

project(loss_func_float32)
pybind11_add_module(loss_func_float32 ${CMAKE_BINARY_DIR}/../pyd_make/loss_func_f32.cpp)

project(layer_node_float32)
pybind11_add_module(layer_node_float32 ${CMAKE_BINARY_DIR}/../pyd_make/layer_node_f32.cpp)

project(actfunc_float32)
pybind11_add_module(actfunc_float32 ${CMAKE_BINARY_DIR}/../pyd_make/actfunc_f32.cpp)

project(reshape_node_float32)
pybind11_add_module(reshape_node_float32 ${CMAKE_BINARY_DIR}/../pyd_make/reshape_node_f32.cpp)

# add_executable(example test.cpp)

# project(Tutorial2)
# pybind11_add_module(Tutorial2 ${CMAKE_BINARY_DIR}/../debug/make_pyd2.cpp)
# target_link_directories(sub PUBLIC D:/anaconda3/envs/pybind_demo/libs)
# target_link_libraries(Tutorial PUBLIC sub)

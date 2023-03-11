#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <set>
#include <vector>
#include <xtensor-python/pyarray.hpp>
#include <xtensor/xarray.hpp>
#include "../src/node/node.hpp"
#include "../src/cal/basic_operation.hpp"
#include "../src/cal/elmt_func.hpp"
#include "../src/datatype.h"
#include"../src/actfunc/LeakyReLU.hpp"
#include"../src/actfunc/ReLU.hpp"
#include"../src/actfunc/sigmoid.hpp"
#include"../src/actfunc/softmax.hpp"

#define FORCE_IMPORT_ARRAY
#define CAL_NODE_FLOAT32 __CAL_NODE__<FLOAT32>
using namespace xt;
namespace py = pybind11;

PYBIND11_MODULE(actfunc_float32, m)
{
    py::class_<LEAKYRELU_NODE_FLOAT32, CAL_NODE_FLOAT32>(m, "leakyrelu_node_float32")
        // .def(py::init<>())
        .def(py::init<FLOAT32>())

        .def_readwrite("pre_node", &LEAKYRELU_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &LEAKYRELU_NODE_FLOAT32::back_node)
        .def_readwrite("if_forward", &LEAKYRELU_NODE_FLOAT32::if_forward)
        .def_readwrite("my_properity", &LEAKYRELU_NODE_FLOAT32::my_properity)
        .def_readwrite("negative_slope",&LEAKYRELU_NODE_FLOAT32::negative_slope)



        .def("SetPreNode", &LEAKYRELU_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", &LEAKYRELU_NODE_FLOAT32::SetBackNode)
        .def("AddPreNode", &LEAKYRELU_NODE_FLOAT32::AddPreNode)
        .def("Forward", &LEAKYRELU_NODE_FLOAT32::Forward)
        .def("Backward", &LEAKYRELU_NODE_FLOAT32::Backward);

    py::class_<RELU_NODE_FLOAT32, CAL_NODE_FLOAT32>(m, "relu_node_float32")
        // .def(py::init<>())
        .def(py::init<>())

        .def_readwrite("pre_node", &RELU_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &RELU_NODE_FLOAT32::back_node)
        .def_readwrite("if_forward", &RELU_NODE_FLOAT32::if_forward)
        .def_readwrite("my_properity", &RELU_NODE_FLOAT32::my_properity)



        .def("SetPreNode", &RELU_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", &RELU_NODE_FLOAT32::SetBackNode)
        .def("AddPreNode", &RELU_NODE_FLOAT32::AddPreNode)
        .def("Forward", &RELU_NODE_FLOAT32::Forward)
        .def("Backward", &RELU_NODE_FLOAT32::Backward);

    py::class_<SIGMOID_NODE_FLOAT32, CAL_NODE_FLOAT32>(m, "sigmoid_node_float32")
        // .def(py::init<>())
        .def(py::init<>())

        .def_readwrite("pre_node", &SIGMOID_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &SIGMOID_NODE_FLOAT32::back_node)
        .def_readwrite("if_forward", &SIGMOID_NODE_FLOAT32::if_forward)
        .def_readwrite("my_properity", &SIGMOID_NODE_FLOAT32::my_properity)



        .def("SetPreNode", &SIGMOID_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", &SIGMOID_NODE_FLOAT32::SetBackNode)
        .def("AddPreNode", &SIGMOID_NODE_FLOAT32::AddPreNode)
        .def("Forward", &SIGMOID_NODE_FLOAT32::Forward)
        .def("Backward", &SIGMOID_NODE_FLOAT32::Backward);

    py::class_<SOFTMAX_NODE_FLOAT32, CAL_NODE_FLOAT32>(m, "softmax_node_float32")
        // .def(py::init<>())
        .def(py::init<>())

        .def_readwrite("pre_node", &SOFTMAX_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &SOFTMAX_NODE_FLOAT32::back_node)
        .def_readwrite("if_forward", &SOFTMAX_NODE_FLOAT32::if_forward)
        .def_readwrite("my_properity", &SOFTMAX_NODE_FLOAT32::my_properity)



        .def("SetPreNode", &SOFTMAX_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", &SOFTMAX_NODE_FLOAT32::SetBackNode)
        .def("AddPreNode", &SOFTMAX_NODE_FLOAT32::AddPreNode)
        .def("Forward", &SOFTMAX_NODE_FLOAT32::Forward)
        .def("Backward", &SOFTMAX_NODE_FLOAT32::Backward);
}
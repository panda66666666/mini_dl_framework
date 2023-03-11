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
#include "../src/layer/linear.hpp"
#include"../src/layer/conv2d.hpp"
#include "../src/layer/pooling.hpp"

#define FORCE_IMPORT_ARRAY
#define CAL_NODE_FLOAT32 __CAL_NODE__<FLOAT32>
using namespace xt;
namespace py = pybind11;

PYBIND11_MODULE(layer_node_float32, m)
{
    py::class_<LINEAR_NODE_FLOAT32, CAL_NODE_FLOAT32>(m, "linear_node_float32")
        // .def(py::init<>())
        .def(py::init<UNSIGNED_INT32,UNSIGNED_INT32,bool>())

        .def_readwrite("pre_node", &LINEAR_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &LINEAR_NODE_FLOAT32::back_node)
        .def_readwrite("if_forward", &LINEAR_NODE_FLOAT32::if_forward)
        .def_readwrite("my_properity", &LINEAR_NODE_FLOAT32::my_properity)
        .def_readwrite("weights", &LINEAR_NODE_FLOAT32::weights)
        .def_readwrite("weights_grad", &LINEAR_NODE_FLOAT32::weights_grad)
        .def_readwrite("bias", &LINEAR_NODE_FLOAT32::bias)
        .def_readwrite("bias_grad", &LINEAR_NODE_FLOAT32::bias_grad)
        .def_readwrite("in_features", &LINEAR_NODE_FLOAT32::in_features)
        .def_readwrite("out_features", &LINEAR_NODE_FLOAT32::out_features)



        .def("SetPreNode", &LINEAR_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", &LINEAR_NODE_FLOAT32::SetBackNode)
        .def("AddPreNode", &LINEAR_NODE_FLOAT32::AddPreNode)
        .def("Forward", &LINEAR_NODE_FLOAT32::Forward)
        .def("Backward", &LINEAR_NODE_FLOAT32::Backward);

    py::class_<CONV2D_NODE_FLOAT32, CAL_NODE_FLOAT32>(m, "conv2d_node_float32")
        // .def(py::init<>())
        .def(py::init<UNSIGNED_INT32, UNSIGNED_INT32, UNSIGNED_INT32, UNSIGNED_INT32, UNSIGNED_INT32,bool>())

        .def_readwrite("pre_node", &CONV2D_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &CONV2D_NODE_FLOAT32::back_node)
        .def_readwrite("if_forward", &CONV2D_NODE_FLOAT32::if_forward)
        .def_readwrite("my_properity", &CONV2D_NODE_FLOAT32::my_properity)
        .def_readwrite("in_channels", &CONV2D_NODE_FLOAT32::in_channels)
        .def_readwrite("out_channel", &CONV2D_NODE_FLOAT32::out_channel)
        .def_readwrite("kernel_size", &CONV2D_NODE_FLOAT32::kernel_size)
        .def_readwrite("stride", &CONV2D_NODE_FLOAT32::stride)
        .def_readwrite("padding", &CONV2D_NODE_FLOAT32::padding)
        .def_readwrite("need_bias", &CONV2D_NODE_FLOAT32::need_bias)
        .def_readwrite("kernel", &CONV2D_NODE_FLOAT32::kernel)
        .def_readwrite("kernel_grad", &CONV2D_NODE_FLOAT32::kernel_grad)
        .def_readwrite("bias", &CONV2D_NODE_FLOAT32::bias)
        .def_readwrite("bias_grad", &CONV2D_NODE_FLOAT32::bias_grad)



        .def("SetPreNode", &CONV2D_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", &CONV2D_NODE_FLOAT32::SetBackNode)
        .def("AddPreNode", &CONV2D_NODE_FLOAT32::AddPreNode)
        .def("Forward", &CONV2D_NODE_FLOAT32::Forward)
        .def("Backward", &CONV2D_NODE_FLOAT32::Backward);

    py::class_<AVGPOOL2D_NODE_FLOAT32, CAL_NODE_FLOAT32>(m, "avgpool2d_node_float32")
        // .def(py::init<>())
        .def(py::init<UNSIGNED_INT32, UNSIGNED_INT32, UNSIGNED_INT32>())

        .def_readwrite("pre_node", &AVGPOOL2D_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &AVGPOOL2D_NODE_FLOAT32::back_node)
        .def_readwrite("if_forward", &AVGPOOL2D_NODE_FLOAT32::if_forward)
        .def_readwrite("my_properity", &AVGPOOL2D_NODE_FLOAT32::my_properity)
        .def_readwrite("kernel_size", &AVGPOOL2D_NODE_FLOAT32::kernel_size)
        .def_readwrite("stride", &AVGPOOL2D_NODE_FLOAT32::stride)
        .def_readwrite("padding", &AVGPOOL2D_NODE_FLOAT32::padding)




        .def("SetPreNode", &AVGPOOL2D_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", &AVGPOOL2D_NODE_FLOAT32::SetBackNode)
        .def("AddPreNode", &AVGPOOL2D_NODE_FLOAT32::AddPreNode)
        .def("Forward", &AVGPOOL2D_NODE_FLOAT32::Forward)
        .def("Backward", &AVGPOOL2D_NODE_FLOAT32::Backward);
}
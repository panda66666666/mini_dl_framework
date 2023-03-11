#include <string>
#include <set>
#include <vector>
#include <xtensor-python/pyarray.hpp>
#include <xtensor/xarray.hpp>
#include <pybind11/stl.h>
#include "../src/node/node.hpp"
#include "../src/cal/basic_operation.hpp"
#include "../src/cal/elmt_func.hpp"
#include "../src/datatype.h"
#include "../src/utils/reshape.hpp"

#define FORCE_IMPORT_ARRAY
#define CAL_NODE_FLOAT32 __CAL_NODE__<FLOAT32>
using namespace xt;
namespace py = pybind11;

PYBIND11_MODULE(reshape_node_float32, m)
{
    py::class_<RESHAPE_NODE_FLOAT32, CAL_NODE_FLOAT32>(m, "reshape_node_float32")
        // .def(py::init<>())
        .def(py::init<std::vector<INT32>>())

        .def_readwrite("pre_node", &RESHAPE_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &RESHAPE_NODE_FLOAT32::back_node)
        .def_readwrite("if_forward", &RESHAPE_NODE_FLOAT32::if_forward)
        .def_readwrite("my_properity", &RESHAPE_NODE_FLOAT32::my_properity)
        .def_readwrite("output_shape", &RESHAPE_NODE_FLOAT32::output_shape)



        .def("SetPreNode", &RESHAPE_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", &RESHAPE_NODE_FLOAT32::SetBackNode)
        .def("AddPreNode", &RESHAPE_NODE_FLOAT32::AddPreNode)
        .def("Forward", &RESHAPE_NODE_FLOAT32::Forward)
        .def("Backward", &RESHAPE_NODE_FLOAT32::Backward);
}
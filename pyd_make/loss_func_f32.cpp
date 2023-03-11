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
#include "../src/lossfunc/mseloss.hpp"
#include "../src/lossfunc/celoss.hpp"

#define FORCE_IMPORT_ARRAY
#define CAL_NODE_FLOAT32 __CAL_NODE__<FLOAT32>

namespace py = pybind11;

PYBIND11_MODULE(loss_func_float32, m)
{
    py::class_<MSELOSS_NODE_FLOAT32, CAL_NODE_FLOAT32>(m, "mseloss_node_float32")
        .def(py::init<>())

        .def_readwrite("pre_node", &MSELOSS_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &MSELOSS_NODE_FLOAT32::back_node)
        .def_readwrite("if_forward", &MSELOSS_NODE_FLOAT32::if_forward)
        .def_readwrite("my_properity", &MSELOSS_NODE_FLOAT32::my_properity)

        .def("SetPreNode", &MSELOSS_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", &MSELOSS_NODE_FLOAT32::SetBackNode)
        .def("AddPreNode", &MSELOSS_NODE_FLOAT32::AddPreNode)
        .def("Forward", &MSELOSS_NODE_FLOAT32::Forward)
        .def("Backward", &MSELOSS_NODE_FLOAT32::Backward);

    py::class_<CELOSS_NODE_FLOAT32, CAL_NODE_FLOAT32>(m, "celoss_node_float32")
        .def(py::init<>())

        .def_readwrite("pre_node", &CELOSS_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &CELOSS_NODE_FLOAT32::back_node)
        .def_readwrite("if_forward", &CELOSS_NODE_FLOAT32::if_forward)
        .def_readwrite("my_properity", &CELOSS_NODE_FLOAT32::my_properity)

        .def("SetPreNode", &CELOSS_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", &CELOSS_NODE_FLOAT32::SetBackNode)
        .def("AddPreNode", &CELOSS_NODE_FLOAT32::AddPreNode)
        .def("Forward", &CELOSS_NODE_FLOAT32::Forward)
        .def("Backward", &CELOSS_NODE_FLOAT32::Backward);
}
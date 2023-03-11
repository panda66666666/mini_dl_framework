#include "../lib/pybind11/include/pybind11/pybind11.h"
#include "../src/node/node.hpp"
#include <vector>
#include "../src/datatype.h"
#include "../lib/xtensor/include/xtensor/xarray.hpp"
#include "../src/utils/cal_map_flow.hpp"
#include <string>
#include <set>
#include <xtensor-python/pyarray.hpp>
#include <pybind11/stl.h>
#define FORCE_IMPORT_ARRAY
using namespace xt;
namespace py = pybind11;

PYBIND11_MODULE(data_node_float32, m)
{
    py::class_<DATA_NODE_FLOAT32>(m, "data_node_float32")
        .def(py::init<>())
        .def(py::init<xarray<FLOAT32> &>())
        .def(py::init<const xarray<FLOAT32> &>())

        .def_readwrite("pre_node", &DATA_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &DATA_NODE_FLOAT32::back_node)
        .def_readwrite("data", &DATA_NODE_FLOAT32::data)
        .def_readwrite("grad", &DATA_NODE_FLOAT32::grad)
        .def_readwrite("if_backward", &DATA_NODE_FLOAT32::if_backward)
        .def_readwrite("require_grad", &DATA_NODE_FLOAT32::require_grad)
        .def_readwrite("my_properity", &DATA_NODE_FLOAT32::my_properity)
        .def_readwrite("my_map", &DATA_NODE_FLOAT32::my_map)

        .def("SetPreNode", &DATA_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", static_cast<void (DATA_NODE_FLOAT32::*)(std::vector<__CAL_NODE__<FLOAT32> *> &)>(&DATA_NODE_FLOAT32::SetBackNode))
        .def("SetBackNode", static_cast<void (DATA_NODE_FLOAT32::*)(const std::vector<__CAL_NODE__<FLOAT32> *> &)>(&DATA_NODE_FLOAT32::SetBackNode))
        .def("AddBackNode", &DATA_NODE_FLOAT32::AddBackNode)
        .def("SetData", static_cast<void (DATA_NODE_FLOAT32::*)(xarray<FLOAT32> &)>(&DATA_NODE_FLOAT32::SetData))
        .def("SetData", static_cast<void (DATA_NODE_FLOAT32::*)(const xarray<FLOAT32> &)>(&DATA_NODE_FLOAT32::SetData))
        .def("SetGrad", static_cast<void (DATA_NODE_FLOAT32::*)(xarray<FLOAT32> &)>(&DATA_NODE_FLOAT32::SetGrad))
        .def("SetGrad", static_cast<void (DATA_NODE_FLOAT32::*)(const xarray<FLOAT32> &)>(&DATA_NODE_FLOAT32::SetGrad))
        .def("Backward", &DATA_NODE_FLOAT32::Backward)
        .def("SetMyMap", &DATA_NODE_FLOAT32::SetMyMap);
    // .def("doit", &DATA_NODE_FLOAT32::doit);

    py::class_<CONST_NODE_FLOAT32, DATA_NODE_FLOAT32>(m, "const_node_float32")
        .def(py::init<>())
        .def(py::init<xarray<FLOAT32> &>())
        .def(py::init<const xarray<FLOAT32> &>())

        .def_readwrite("pre_node", &CONST_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &CONST_NODE_FLOAT32::back_node)
        .def_readwrite("data", &CONST_NODE_FLOAT32::data)
        .def_readwrite("grad", &CONST_NODE_FLOAT32::grad)
        .def_readwrite("if_backward", &CONST_NODE_FLOAT32::if_backward)
        .def_readwrite("require_grad", &CONST_NODE_FLOAT32::require_grad)
        .def_readwrite("my_properity", &CONST_NODE_FLOAT32::my_properity)
        .def_readwrite("my_map", &CONST_NODE_FLOAT32::my_map)

        .def("SetPreNode", &CONST_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", static_cast<void (CONST_NODE_FLOAT32::*)(std::vector<__CAL_NODE__<FLOAT32> *> &)>(&CONST_NODE_FLOAT32::SetBackNode))
        .def("SetBackNode", static_cast<void (CONST_NODE_FLOAT32::*)(const std::vector<__CAL_NODE__<FLOAT32> *> &)>(&CONST_NODE_FLOAT32::SetBackNode))
        .def("AddBackNode", &CONST_NODE_FLOAT32::AddBackNode)
        .def("SetData", static_cast<void (CONST_NODE_FLOAT32::*)(xarray<FLOAT32> &)>(&CONST_NODE_FLOAT32::SetData))
        .def("SetData", static_cast<void (CONST_NODE_FLOAT32::*)(const xarray<FLOAT32> &)>(&CONST_NODE_FLOAT32::SetData))
        .def("SetGrad", static_cast<void (CONST_NODE_FLOAT32::*)(xarray<FLOAT32> &)>(&CONST_NODE_FLOAT32::SetGrad))
        .def("SetGrad", static_cast<void (CONST_NODE_FLOAT32::*)(const xarray<FLOAT32> &)>(&CONST_NODE_FLOAT32::SetGrad))
        .def("Backward", &CONST_NODE_FLOAT32::Backward)
        .def("SetMyMap", &CONST_NODE_FLOAT32::SetMyMap);
    // .def("doit", &CONST_NODE_FLOAT32::doit);
}

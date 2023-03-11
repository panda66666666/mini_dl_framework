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
#include "../src/utils/cal_map_flow.hpp"
#include "../src/datatype.h"

#define FORCE_IMPORT_ARRAY
namespace py = pybind11;

PYBIND11_MODULE(cal_map_flow_float32, m)
{
    py::class_<CAL_MAP_FLOAT32>(m, "cal_map_float32")
        .def(py::init<>())
        .def(py::init<__DATA_NODE__<FLOAT32> *, __DATA_NODE__<FLOAT32> *>())
        .def(py::init<std::vector<__DATA_NODE__<FLOAT32> *> &, std::vector<__DATA_NODE__<FLOAT32> *> &>())

        .def_readwrite("begin_node_list", &CAL_MAP_FLOAT32::begin_node_list)
        .def_readwrite("end_node_list", &CAL_MAP_FLOAT32::end_node_list)
        .def_readwrite("data_node_list", &CAL_MAP_FLOAT32::data_node_list)
        .def_readwrite("para_node_list", &CAL_MAP_FLOAT32::para_node_list)

        .def("AddBeginNode", &CAL_MAP_FLOAT32::AddBeginNode)
        .def("AddEndNode", &CAL_MAP_FLOAT32::AddEndNode)
        .def("AddDataNode", &CAL_MAP_FLOAT32::AddDataNode)
        .def("AddParaNode", &CAL_MAP_FLOAT32::AddParaNode)
        .def("Forward", &CAL_MAP_FLOAT32::Forward)
        .def("Backward", &CAL_MAP_FLOAT32::Backward)
        .def("SetGradZero", &CAL_MAP_FLOAT32::SetGradZero)
        .def("UpdatePara", &CAL_MAP_FLOAT32::UpdatPara);
}
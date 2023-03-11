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

#define FORCE_IMPORT_ARRAY
#define CAL_NODE_FLOAT32 __CAL_NODE__<FLOAT32>

namespace py = pybind11;

PYBIND11_MODULE(opt_node_float32, m)
{

    py::class_<CAL_NODE_FLOAT32, Py_cal_node_float32>(m, "cal_node_float32")
        .def(py::init<>())

        .def_readwrite("pre_node", &CAL_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &CAL_NODE_FLOAT32::back_node)
        .def_readwrite("if_forward", &CAL_NODE_FLOAT32::if_forward)
        .def_readwrite("my_properity", &CAL_NODE_FLOAT32::my_properity)

        .def("SetPreNode", &CAL_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", &CAL_NODE_FLOAT32::SetBackNode)
        .def("AddPreNode", &CAL_NODE_FLOAT32::AddPreNode)
        .def("Forward", &CAL_NODE_FLOAT32::Forward)
        .def("Backward", &CAL_NODE_FLOAT32::Backward);

    /*基础运算算子绑定
    包括：+，-，*，/*/
    py::class_<ADD_NODE_FLOAT32, CAL_NODE_FLOAT32>(m, "add_node_float32")
        .def(py::init<>())

        .def_readwrite("pre_node", &ADD_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &ADD_NODE_FLOAT32::back_node)
        .def_readwrite("if_forward", &ADD_NODE_FLOAT32::if_forward)
        .def_readwrite("my_properity", &ADD_NODE_FLOAT32::my_properity)

        .def("SetPreNode", &ADD_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", &ADD_NODE_FLOAT32::SetBackNode)
        .def("AddPreNode", &ADD_NODE_FLOAT32::AddPreNode)
        .def("Forward", &ADD_NODE_FLOAT32::Forward)
        .def("Backward", &ADD_NODE_FLOAT32::Backward);

    py::class_<SUB_NODE_FLOAT32, CAL_NODE_FLOAT32>(m, "sub_node_float32")
        .def(py::init<>())

        .def_readwrite("pre_node", &SUB_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &SUB_NODE_FLOAT32::back_node)
        .def_readwrite("if_forward", &SUB_NODE_FLOAT32::if_forward)
        .def_readwrite("my_properity", &SUB_NODE_FLOAT32::my_properity)

        .def("SetPreNode", &SUB_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", &SUB_NODE_FLOAT32::SetBackNode)
        .def("AddPreNode", &SUB_NODE_FLOAT32::AddPreNode)
        .def("Forward", &SUB_NODE_FLOAT32::Forward)
        .def("Backward", &SUB_NODE_FLOAT32::Backward);

    py::class_<MUL_NODE_FLOAT32, CAL_NODE_FLOAT32>(m, "mul_node_float32")
        .def(py::init<>())

        .def_readwrite("pre_node", &MUL_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &MUL_NODE_FLOAT32::back_node)
        .def_readwrite("if_forward", &MUL_NODE_FLOAT32::if_forward)
        .def_readwrite("my_properity", &MUL_NODE_FLOAT32::my_properity)

        .def("SetPreNode", &MUL_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", &MUL_NODE_FLOAT32::SetBackNode)
        .def("AddPreNode", &MUL_NODE_FLOAT32::AddPreNode)
        .def("Forward", &MUL_NODE_FLOAT32::Forward)
        .def("Backward", &MUL_NODE_FLOAT32::Backward);

    py::class_<DIV_NODE_FLOAT32, CAL_NODE_FLOAT32>(m, "div_node_float32")
        .def(py::init<>())

        .def_readwrite("pre_node", &DIV_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &DIV_NODE_FLOAT32::back_node)
        .def_readwrite("if_forward", &DIV_NODE_FLOAT32::if_forward)
        .def_readwrite("my_properity", &DIV_NODE_FLOAT32::my_properity)

        .def("SetPreNode", &DIV_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", &DIV_NODE_FLOAT32::SetBackNode)
        .def("AddPreNode", &DIV_NODE_FLOAT32::AddPreNode)
        .def("Forward", &DIV_NODE_FLOAT32::Forward)
        .def("Backward", &DIV_NODE_FLOAT32::Backward);

    /*初等函数算子绑定，
    包括：幂指函数，自然对数指数函数，对数函数，正余弦函数，正切函数*/
    py::class_<POWER_NODE_FLOAT32, CAL_NODE_FLOAT32>(m, "power_node_float32")
        .def(py::init<>())

        .def_readwrite("pre_node", &POWER_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &POWER_NODE_FLOAT32::back_node)
        .def_readwrite("if_forward", &POWER_NODE_FLOAT32::if_forward)
        .def_readwrite("my_properity", &POWER_NODE_FLOAT32::my_properity)

        .def("SetPreNode", &POWER_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", &POWER_NODE_FLOAT32::SetBackNode)
        .def("AddPreNode", &POWER_NODE_FLOAT32::AddPreNode)
        .def("Forward", &POWER_NODE_FLOAT32::Forward)
        .def("Backward", &POWER_NODE_FLOAT32::Backward);

    py::class_<EXP_NODE_FLOAT32, CAL_NODE_FLOAT32>(m, "exp_node_float32")
        .def(py::init<>())

        .def_readwrite("pre_node", &EXP_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &EXP_NODE_FLOAT32::back_node)
        .def_readwrite("if_forward", &EXP_NODE_FLOAT32::if_forward)
        .def_readwrite("my_properity", &EXP_NODE_FLOAT32::my_properity)

        .def("SetPreNode", &EXP_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", &EXP_NODE_FLOAT32::SetBackNode)
        .def("AddPreNode", &EXP_NODE_FLOAT32::AddPreNode)
        .def("Forward", &EXP_NODE_FLOAT32::Forward)
        .def("Backward", &EXP_NODE_FLOAT32::Backward);

    py::class_<LN_NODE_FLOAT32, CAL_NODE_FLOAT32>(m, "ln_node_float32")
        .def(py::init<>())

        .def_readwrite("pre_node", &LN_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &LN_NODE_FLOAT32::back_node)
        .def_readwrite("if_forward", &LN_NODE_FLOAT32::if_forward)
        .def_readwrite("my_properity", &LN_NODE_FLOAT32::my_properity)

        .def("SetPreNode", &LN_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", &LN_NODE_FLOAT32::SetBackNode)
        .def("AddPreNode", &LN_NODE_FLOAT32::AddPreNode)
        .def("Forward", &LN_NODE_FLOAT32::Forward)
        .def("Backward", &LN_NODE_FLOAT32::Backward);

    py::class_<SIN_NODE_FLOAT32, CAL_NODE_FLOAT32>(m, "sin_node_float32")
        .def(py::init<>())

        .def_readwrite("pre_node", &SIN_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &SIN_NODE_FLOAT32::back_node)
        .def_readwrite("if_forward", &SIN_NODE_FLOAT32::if_forward)
        .def_readwrite("my_properity", &SIN_NODE_FLOAT32::my_properity)

        .def("SetPreNode", &SIN_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", &SIN_NODE_FLOAT32::SetBackNode)
        .def("AddPreNode", &SIN_NODE_FLOAT32::AddPreNode)
        .def("Forward", &SIN_NODE_FLOAT32::Forward)
        .def("Backward", &SIN_NODE_FLOAT32::Backward);

    py::class_<COS_NODE_FLOAT32, CAL_NODE_FLOAT32>(m, "cos_node_float32")
        .def(py::init<>())

        .def_readwrite("pre_node", &COS_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &COS_NODE_FLOAT32::back_node)
        .def_readwrite("if_forward", &COS_NODE_FLOAT32::if_forward)
        .def_readwrite("my_properity", &COS_NODE_FLOAT32::my_properity)

        .def("SetPreNode", &COS_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", &COS_NODE_FLOAT32::SetBackNode)
        .def("AddPreNode", &COS_NODE_FLOAT32::AddPreNode)
        .def("Forward", &COS_NODE_FLOAT32::Forward)
        .def("Backward", &COS_NODE_FLOAT32::Backward);

    py::class_<TAN_NODE_FLOAT32, CAL_NODE_FLOAT32>(m, "tan_node_float32")
        .def(py::init<>())

        .def_readwrite("pre_node", &TAN_NODE_FLOAT32::pre_node)
        .def_readwrite("back_node", &TAN_NODE_FLOAT32::back_node)
        .def_readwrite("if_forward", &TAN_NODE_FLOAT32::if_forward)
        .def_readwrite("my_properity", &TAN_NODE_FLOAT32::my_properity)

        .def("SetPreNode", &TAN_NODE_FLOAT32::SetPreNode)
        .def("SetBackNode", &TAN_NODE_FLOAT32::SetBackNode)
        .def("AddPreNode", &TAN_NODE_FLOAT32::AddPreNode)
        .def("Forward", &TAN_NODE_FLOAT32::Forward)
        .def("Backward", &TAN_NODE_FLOAT32::Backward);

    // py::class_<MSELOSS_NODE_FLOAT32, CAL_NODE_FLOAT32>(m, "mseloss_node_float32")
    //     .def(py::init<>())

    //     .def_readwrite("pre_node", &MSELOSS_NODE_FLOAT32::pre_node)
    //     .def_readwrite("back_node", &MSELOSS_NODE_FLOAT32::back_node)
    //     .def_readwrite("if_forward", &MSELOSS_NODE_FLOAT32::if_forward)
    //     .def_readwrite("my_properity", &MSELOSS_NODE_FLOAT32::my_properity)

    //     .def("SetPreNode", &MSELOSS_NODE_FLOAT32::SetPreNode)
    //     .def("SetBackNode", &MSELOSS_NODE_FLOAT32::SetBackNode)
    //     .def("AddPreNode", &MSELOSS_NODE_FLOAT32::AddPreNode)
    //     .def("Forward", &MSELOSS_NODE_FLOAT32::Forward)
    //     .def("Backward", &MSELOSS_NODE_FLOAT32::Backward);
}

import sys
sys.path.append('.')
from Pypanda.dyc_lib.data_node_float32 import data_node_float32, const_node_float32
from Pypanda.dyc_lib.opt_node_float32 import add_node_float32, mul_node_float32, sub_node_float32, div_node_float32, power_node_float32, exp_node_float32, ln_node_float32, sin_node_float32, cos_node_float32, tan_node_float32
from Pypanda.dyc_lib.cal_map_flow_float32 import cal_map_float32
from Pypanda.dyc_lib.loss_func_float32 import mseloss_node_float32

class data_node():
    def __init__(self, c_node):
        self.node = c_node
        self.pre_node = []
        self.back_node = []

    def SetPreNode(self, py_node):
        self.node.SetPreNode(py_node.node)
        self.pre_node = py_node

    def SetBackNode(self, py_node: list):
        self.node.SetBackNode([_.node for _ in py_node])
        self.back_node = py_node

    def AddBackNode(self, py_node):
        # testa=data_node_float32()
        # testb=add_node_float32()

        # testa.AddBackNode(testb)
        self.node.AddBackNode(py_node.node)
        self.back_node.append(py_node)


class cal_node():
    def __init__(self, c_node):
        self.node = c_node
        self.pre_node = []
        self.back_node = []

    def SetPreNode(self, py_node: list):
        self.node.SetPreNode([_.node for _ in py_node])
        self.pre_node = py_node

    def SetBackNode(self, py_node):
        self.node.SetBackNode(py_node.node)
        self.back_node = py_node

    def AddPreNode(self, py_node):
        self.node.AddPreNode(py_node.node)
        self.pre_node.append(py_node)
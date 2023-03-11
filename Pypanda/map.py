import sys
sys.path.append('.')
from Pypanda.variable import Variable
from Pypanda.dyc_lib.data_node_float32 import data_node_float32, const_node_float32
from Pypanda.dyc_lib.opt_node_float32 import add_node_float32, mul_node_float32, sub_node_float32, div_node_float32, power_node_float32, exp_node_float32, ln_node_float32, sin_node_float32, cos_node_float32, tan_node_float32
from Pypanda.dyc_lib.cal_map_flow_float32 import cal_map_float32
from Pypanda.dyc_lib.loss_func_float32 import mseloss_node_float32

class map():
    def __init__(self):
        self.map = cal_map_float32()

    def AddBeginNode(self, py_node):
        self.map.AddBeginNode(py_node.node)

    def AddEndNode(self, py_node):
        self.map.AddEndNode(py_node.node)

    def AddDataNode(self, py_node):
        self.map.AddDataNode(py_node.node)

    def AddParaNode(self, py_node):
        self.map.AddParaNode(py_node.node)

    def Forward(self):
        self.map.Forward()

    def Backward(self):
        self.map.Backward()

    def SetGradZero(self):
        self.map.SetGradZero()
    
    def UpdatePara(self,eta):
        self.map.UpdatePara(eta)
    
    def __lshift__(self,other:Variable):
        self.map.AddEndNode(other.py_node.node)
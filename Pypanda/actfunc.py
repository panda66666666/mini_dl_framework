import sys

sys.path.append('.')
from Pypanda.dyc_lib.data_node_float32 import data_node_float32, const_node_float32
from Pypanda.dyc_lib.opt_node_float32 import add_node_float32, mul_node_float32, sub_node_float32, div_node_float32, power_node_float32, exp_node_float32, ln_node_float32, sin_node_float32, cos_node_float32, tan_node_float32
from Pypanda.dyc_lib.cal_map_flow_float32 import cal_map_float32
from Pypanda.dyc_lib.loss_func_float32 import mseloss_node_float32, celoss_node_float32
# from Pypanda.dyc_lib.layer_node_float32 import linear_node_float32
from Pypanda.dyc_lib.actfunc_float32 import leakyrelu_node_float32,relu_node_float32,sigmoid_node_float32,softmax_node_float32
from Pypanda.variable import *



def LeakyReLU(node):
    #搭建数据流图
    assert isinstance(node, (Variable))
    res = Variable()
    opt = cal_node(leakyrelu_node_float32())

    node.py_node.AddBackNode(opt)
    opt.AddPreNode(node.py_node)
    opt.SetBackNode(res.py_node)
    res.py_node.SetPreNode(opt)

    #搭建节点和数据流图的关系
    if node.py_node.node.my_map != None:
        res.py_node.node.SetMyMap(node.py_node.node.my_map)
        node.py_node.node.my_map.AddDataNode(res.py_node.node)

    return res

def ReLU(node):
    #搭建数据流图
    assert isinstance(node, (Variable))
    res = Variable()
    opt = cal_node(relu_node_float32())

    node.py_node.AddBackNode(opt)
    opt.AddPreNode(node.py_node)
    opt.SetBackNode(res.py_node)
    res.py_node.SetPreNode(opt)

    #搭建节点和数据流图的关系
    if node.py_node.node.my_map != None:
        res.py_node.node.SetMyMap(node.py_node.node.my_map)
        node.py_node.node.my_map.AddDataNode(res.py_node.node)

    return res


def sigmoid(node):
    #搭建数据流图
    assert isinstance(node, (Variable))
    res = Variable()
    opt = cal_node(sigmoid_node_float32())

    node.py_node.AddBackNode(opt)
    opt.AddPreNode(node.py_node)
    opt.SetBackNode(res.py_node)
    res.py_node.SetPreNode(opt)

    #搭建节点和数据流图的关系
    if node.py_node.node.my_map != None:
        res.py_node.node.SetMyMap(node.py_node.node.my_map)
        node.py_node.node.my_map.AddDataNode(res.py_node.node)

    return res



def softmax(node):
    #搭建数据流图
    assert isinstance(node, (Variable))
    res = Variable()
    opt = cal_node(softmax_node_float32())

    node.py_node.AddBackNode(opt)
    opt.AddPreNode(node.py_node)
    opt.SetBackNode(res.py_node)
    res.py_node.SetPreNode(opt)

    #搭建节点和数据流图的关系
    if node.py_node.node.my_map != None:
        res.py_node.node.SetMyMap(node.py_node.node.my_map)
        node.py_node.node.my_map.AddDataNode(res.py_node.node)

    return res



import sys


sys.path.append('.')
from Pypanda.dyc_lib.data_node_float32 import data_node_float32, const_node_float32
from Pypanda.dyc_lib.opt_node_float32 import add_node_float32, mul_node_float32, sub_node_float32, div_node_float32, power_node_float32, exp_node_float32, ln_node_float32, sin_node_float32, cos_node_float32, tan_node_float32
from Pypanda.dyc_lib.cal_map_flow_float32 import cal_map_float32
from Pypanda.dyc_lib.loss_func_float32 import mseloss_node_float32, celoss_node_float32
# from Pypanda.dyc_lib.layer_node_float32 import linear_node_float32
from Pypanda.dyc_lib.layer_node_float32 import linear_node_float32,conv2d_node_float32,avgpool2d_node_float32
from Pypanda.variable import *


class Linear():
    def __init__(self, in_features, out_features, need_bias=True) -> None:

        self.linear_node = linear_node_float32(in_features, out_features,
                                               need_bias)
        self.need_bias = need_bias

        self.in_features = in_features
        self.out_features = out_features

    def GetW(self):
        return self.linear_node.weights

    def GetWGrad(self):
        return self.linear_node.weights_grad

    def GetBias(self):
        if self.need_bias:
            return self.linear_node.bias
        print("the Linear layer has no bias")

    def GetBiasGrad(self):
        if self.need_bias:
            return self.linear_node.bias_grad
        print("the Linear layer has no bias")

    def __call__(self, node):
        #搭建数据流图
        assert isinstance(node, (Variable))
        res = Variable()
        opt = cal_node(self.linear_node)

        node.py_node.AddBackNode(opt)
        opt.AddPreNode(node.py_node)
        opt.SetBackNode(res.py_node)
        res.py_node.SetPreNode(opt)

        #搭建节点和数据流图的关系
        if node.py_node.node.my_map != None:
            res.py_node.node.SetMyMap(node.py_node.node.my_map)
            node.py_node.node.my_map.AddDataNode(res.py_node.node)

        return res

class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size,stride=1,padding=0,need_bias=True) -> None:

        self.conv2d_node = conv2d_node_float32(in_channels, out_channels,
                                               kernel_size,stride,padding,need_bias)


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.need_bias = need_bias
        

    def GetW(self):
        return self.conv2d_node.kernel

    def GetWGrad(self):
        return self.conv2d_node.kernel_grad

    def GetBias(self):
        if self.need_bias:
            return self.conv2d_node.bias
        print("the Conv2d layer has no bias")

    def GetBiasGrad(self):
        if self.need_bias:
            return self.conv2d_node.bias_grad
        print("the Conv2d layer has no bias")

    def __call__(self, node):
        #搭建数据流图
        assert isinstance(node, (Variable))
        res = Variable()
        opt = cal_node(self.conv2d_node)

        node.py_node.AddBackNode(opt)
        opt.AddPreNode(node.py_node)
        opt.SetBackNode(res.py_node)
        res.py_node.SetPreNode(opt)

        #搭建节点和数据流图的关系
        if node.py_node.node.my_map != None:
            res.py_node.node.SetMyMap(node.py_node.node.my_map)
            node.py_node.node.my_map.AddDataNode(res.py_node.node)

        return res



class AvgPool2d():
    def __init__(self,kernel_size,stride=1,padding=0) -> None:

        self.avgpool2d_node = avgpool2d_node_float32(kernel_size, stride,padding)

        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding


    def __call__(self, node):
        #搭建数据流图
        assert isinstance(node, (Variable))
        res = Variable()
        opt = cal_node(self.avgpool2d_node)

        node.py_node.AddBackNode(opt)
        opt.AddPreNode(node.py_node)
        opt.SetBackNode(res.py_node)
        res.py_node.SetPreNode(opt)

        #搭建节点和数据流图的关系
        if node.py_node.node.my_map != None:
            res.py_node.node.SetMyMap(node.py_node.node.my_map)
            node.py_node.node.my_map.AddDataNode(res.py_node.node)

        return res

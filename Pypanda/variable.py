import sys
import random
import numpy as np
sys.path.append('.')
#sys.path.append('D:\\Users\\Hi\\Desktop\\source_code\\workspace_vs')
from Pypanda.dyc_lib.data_node_float32 import data_node_float32, const_node_float32
from Pypanda.dyc_lib.opt_node_float32 import add_node_float32, mul_node_float32, sub_node_float32, div_node_float32, power_node_float32, exp_node_float32, ln_node_float32, sin_node_float32, cos_node_float32, tan_node_float32
from Pypanda.dyc_lib.cal_map_flow_float32 import cal_map_float32
from Pypanda.dyc_lib.loss_func_float32 import mseloss_node_float32
from Pypanda.node import *
from Pypanda.map import*


# a = np.array([1, 2])
# print(type(a))
# # print(isinstance(99.2,(np.ndarray,int,float)))
# test = data_node_float32()
# print(isinstance(test, (data_node_float32, const_node_float32)))

class Variable():
    def __init__(self, data=None, need_grad: bool = True):
        if isinstance(data,(int,float)):
            if need_grad:
                if data != None:
                    self.py_node = data_node(data_node_float32(data))
                else:
                    self.py_node = data_node(data_node_float32())
            else:
                if data != None:
                    self.py_node = data_node(const_node_float32(data))
                else:
                    self.py_node = data_node(const_node_float32())

        elif isinstance(data,(np.ndarray)):
            if need_grad:
                if data.any() != None:
                    self.py_node = data_node(data_node_float32(data))
                else:
                    self.py_node = data_node(data_node_float32())
            else:
                if data.any() != None:
                    self.py_node = data_node(const_node_float32(data))
                else:
                    self.py_node = data_node(const_node_float32())
        
        else:
            self.py_node=data_node(data_node_float32())

    def GetData(self):
        return self.py_node.node.data
    
    def GetGrad(self):
        return self.py_node.node.grad
    
    def SetData(self,num):
        assert isinstance(num, (int, float, np.ndarray))
        self.py_node.node.SetData(num)

    def __rshift__(self, map: map):
        self.py_node.node.SetMyMap(map.map)
        map.AddBeginNode(self.py_node)
        map.AddDataNode(self.py_node)

    def __neg__(self):
        self.py_node.node.data *= -1

    def __add__(self, other):
        assert isinstance(other, (int, float, np.ndarray, Variable))

        opt = cal_node(add_node_float32())
        res = Variable()

        if isinstance(other, (int, float, np.ndarray)):
            tem = Variable(other)

            #搭建数据流图
            tem.py_node.AddBackNode(opt)
            self.py_node.AddBackNode(opt)
            opt.AddPreNode(self.py_node)
            opt.AddPreNode(tem.py_node)

            opt.SetBackNode(res.py_node)
            res.py_node.SetPreNode(opt)

            #搭建节点和数据流图的关系
            if self.py_node.node.my_map != None:
                self.py_node.node.my_map.AddBeginNode(tem.py_node.node)
                self.py_node.node.my_map.AddDataNode(tem.py_node.node)
                self.py_node.node.my_map.AddDataNode(res.py_node.node)

                tem.py_node.node.SetMyMap(self.py_node.node.my_map)
                res.py_node.node.SetMyMap(self.py_node.node.my_map)

            return res
        else:
            m1, m2 = self.py_node.node.my_map, other.py_node.node.my_map
            if m1 == None and m2 != None:
                #搭建节点和数据流图的关系
                other.py_node.node.my_map.AddBeginNode(self.py_node.node)
                other.py_node.node.my_map.AddDataNode(self.py_node.node)
                other.py_node.node.my_map.AddDataNode(res.py_node.node)

                self.py_node.node.SetMyMap(other.py_node.node.my_map)
                res.py_node.node.SetMyMap(other.py_node.node.my_map)

                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(self.py_node)
                opt.AddPreNode(other.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            elif m1 != None and m2 == None:
                #搭建节点和数据流图的关系
                self.py_node.node.my_map.AddBeginNode(other.py_node.node)
                self.py_node.node.my_map.AddDataNode(other.py_node.node)
                self.py_node.node.my_map.AddDataNode(res.py_node.node)

                other.py_node.node.SetMyMap(self.py_node.node.my_map)
                res.py_node.node.SetMyMap(self.py_node.node.my_map)

                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(self.py_node)
                opt.AddPreNode(other.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            elif m1 == None and m2 == None:
                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(self.py_node)
                opt.AddPreNode(other.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            else:
                if self.py_node.node.my_map == other.py_node.node.my_map:
                    #搭建节点和数据流图关系
                    self.py_node.node.my_map.AddDataNode(res.py_node.node)
                    res.py_node.node.SetMyMap(self.py_node.node.my_map)

                    #搭建数据流图
                    other.py_node.AddBackNode(opt)
                    self.py_node.AddBackNode(opt)
                    opt.AddPreNode(self.py_node)
                    opt.AddPreNode(other.py_node)

                    opt.SetBackNode(res.py_node)
                    res.py_node.SetPreNode(opt)
                else:
                    print('节点错误：暂时不支持多数据流图混合编程！！')

            return res

    def __radd__(self, other):
        assert isinstance(other, (int, float, np.ndarray, Variable))

        opt = cal_node(add_node_float32())
        res = Variable()

        if isinstance(other, (int, float, np.ndarray)):
            tem = Variable(other)

            #搭建数据流图
            tem.py_node.AddBackNode(opt)
            self.py_node.AddBackNode(opt)
            opt.AddPreNode(tem.py_node)
            opt.AddPreNode(self.py_node)

            opt.SetBackNode(res.py_node)
            res.py_node.SetPreNode(opt)

            #搭建节点和数据流图的关系
            if self.py_node.node.my_map != None:
                self.py_node.node.my_map.AddBeginNode(tem.py_node.node)
                self.py_node.node.my_map.AddDataNode(tem.py_node.node)
                self.py_node.node.my_map.AddDataNode(res.py_node.node)

                tem.py_node.node.SetMyMap(self.py_node.node.my_map)
                res.py_node.node.SetMyMap(self.py_node.node.my_map)

            return res
        else:
            m1, m2 = self.py_node.node.my_map, other.py_node.node.my_map
            if m1 == None and m2 != None:
                #搭建节点和数据流图的关系
                other.py_node.node.my_map.AddBeginNode(self.py_node.node)
                other.py_node.node.my_map.AddDataNode(self.py_node.node)
                other.py_node.node.my_map.AddDataNode(res.py_node.node)

                self.py_node.node.SetMyMap(other.py_node.node.my_map)
                res.py_node.node.SetMyMap(other.py_node.node.my_map)

                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(other.py_node)
                opt.AddPreNode(self.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            elif m1 != None and m2 == None:
                #搭建节点和数据流图的关系
                self.py_node.node.my_map.AddBeginNode(other.py_node.node)
                self.py_node.node.my_map.AddDataNode(other.py_node.node)
                self.py_node.node.my_map.AddDataNode(res.py_node.node)

                other.py_node.node.SetMyMap(self.py_node.node.my_map)
                res.py_node.node.SetMyMap(self.py_node.node.my_map)

                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(other.py_node)
                opt.AddPreNode(self.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            elif m1 == None and m2 == None:

                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(other.py_node)
                opt.AddPreNode(self.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            else:
                if self.py_node.node.my_map == other.py_node.node.my_map:
                    #搭建节点和数据流图关系
                    self.py_node.node.my_map.AddDataNode(res.py_node.node)
                    res.py_node.node.SetMyMap(self.py_node.node.my_map)

                    #搭建数据流图
                    other.py_node.AddBackNode(opt)
                    self.py_node.AddBackNode(opt)
                    opt.AddPreNode(other.py_node)
                    opt.AddPreNode(self.py_node)

                    opt.SetBackNode(res.py_node)
                    res.py_node.SetPreNode(opt)
                else:
                    print('节点错误：暂时不支持多数据流图混合编程！！')

            return res

    def __sub__(self, other):
        assert isinstance(other, (int, float, np.ndarray, Variable))

        opt = cal_node(sub_node_float32())
        res = Variable()

        if isinstance(other, (int, float, np.ndarray)):
            tem = Variable(other)

            #搭建数据流图
            tem.py_node.AddBackNode(opt)
            self.py_node.AddBackNode(opt)
            opt.AddPreNode(self.py_node)
            opt.AddPreNode(tem.py_node)

            opt.SetBackNode(res.py_node)
            res.py_node.SetPreNode(opt)

            #搭建节点和数据流图的关系
            if self.py_node.node.my_map != None:
                self.py_node.node.my_map.AddBeginNode(tem.py_node.node)
                self.py_node.node.my_map.AddDataNode(tem.py_node.node)
                self.py_node.node.my_map.AddDataNode(res.py_node.node)

                tem.py_node.node.SetMyMap(self.py_node.node.my_map)
                res.py_node.node.SetMyMap(self.py_node.node.my_map)

            return res
        else:
            m1, m2 = self.py_node.node.my_map, other.py_node.node.my_map
            if m1 == None and m2 != None:
                #搭建节点和数据流图的关系
                other.py_node.node.my_map.AddBeginNode(self.py_node.node)
                other.py_node.node.my_map.AddDataNode(self.py_node.node)
                other.py_node.node.my_map.AddDataNode(res.py_node.node)

                self.py_node.node.SetMyMap(other.py_node.node.my_map)
                res.py_node.node.SetMyMap(other.py_node.node.my_map)

                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(self.py_node)
                opt.AddPreNode(other.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            elif m1 != None and m2 == None:
                #搭建节点和数据流图的关系
                self.py_node.node.my_map.AddBeginNode(other.py_node.node)
                self.py_node.node.my_map.AddDataNode(other.py_node.node)
                self.py_node.node.my_map.AddDataNode(res.py_node.node)

                other.py_node.node.SetMyMap(self.py_node.node.my_map)
                res.py_node.node.SetMyMap(self.py_node.node.my_map)

                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(self.py_node)
                opt.AddPreNode(other.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            elif m1 == None and m2 == None:
                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(self.py_node)
                opt.AddPreNode(other.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            else:
                if self.py_node.node.my_map == other.py_node.node.my_map:
                    #搭建节点和数据流图关系
                    self.py_node.node.my_map.AddDataNode(res.py_node.node)
                    res.py_node.node.SetMyMap(self.py_node.node.my_map)

                    #搭建数据流图
                    other.py_node.AddBackNode(opt)
                    self.py_node.AddBackNode(opt)
                    opt.AddPreNode(self.py_node)
                    opt.AddPreNode(other.py_node)

                    opt.SetBackNode(res.py_node)
                    res.py_node.SetPreNode(opt)
                else:
                    print('节点错误：暂时不支持多数据流图混合编程！！')

            return res

    def __rsub__(self, other):
        assert isinstance(other, (int, float, np.ndarray, Variable))

        opt = cal_node(sub_node_float32())
        res = Variable()

        if isinstance(other, (int, float, np.ndarray)):
            tem = Variable(other)

            #搭建数据流图
            tem.py_node.AddBackNode(opt)
            self.py_node.AddBackNode(opt)
            opt.AddPreNode(tem.py_node)
            opt.AddPreNode(self.py_node)

            opt.SetBackNode(res.py_node)
            res.py_node.SetPreNode(opt)

            #搭建节点和数据流图的关系
            if self.py_node.node.my_map != None:
                self.py_node.node.my_map.AddBeginNode(tem.py_node.node)
                self.py_node.node.my_map.AddDataNode(tem.py_node.node)
                self.py_node.node.my_map.AddDataNode(res.py_node.node)

                tem.py_node.node.SetMyMap(self.py_node.node.my_map)
                res.py_node.node.SetMyMap(self.py_node.node.my_map)

            return res
        else:
            m1, m2 = self.py_node.node.my_map, other.py_node.node.my_map
            if m1 == None and m2 != None:
                #搭建节点和数据流图的关系
                other.py_node.node.my_map.AddBeginNode(self.py_node.node)
                other.py_node.node.my_map.AddDataNode(self.py_node.node)
                other.py_node.node.my_map.AddDataNode(res.py_node.node)

                self.py_node.node.SetMyMap(other.py_node.node.my_map)
                res.py_node.node.SetMyMap(other.py_node.node.my_map)

                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(other.py_node)
                opt.AddPreNode(self.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            elif m1 != None and m2 == None:
                #搭建节点和数据流图的关系
                self.py_node.node.my_map.AddBeginNode(other.py_node.node)
                self.py_node.node.my_map.AddDataNode(other.py_node.node)
                self.py_node.node.my_map.AddDataNode(res.py_node.node)

                other.py_node.node.SetMyMap(self.py_node.node.my_map)
                res.py_node.node.SetMyMap(self.py_node.node.my_map)

                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(other.py_node)
                opt.AddPreNode(self.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            elif m1 == None and m2 == None:
                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(other.py_node)
                opt.AddPreNode(self.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            else:
                if self.py_node.node.my_map == other.py_node.node.my_map:
                    #搭建节点和数据流图关系
                    self.py_node.node.my_map.AddDataNode(res.py_node.node)
                    res.py_node.node.SetMyMap(self.py_node.node.my_map)

                    #搭建数据流图
                    other.py_node.AddBackNode(opt)
                    self.py_node.AddBackNode(opt)
                    opt.AddPreNode(other.py_node)
                    opt.AddPreNode(self.py_node)

                    opt.SetBackNode(res.py_node)
                    res.py_node.SetPreNode(opt)
                else:
                    print('节点错误：暂时不支持多数据流图混合编程！！')

            return res

    def __mul__(self, other):
        assert isinstance(other, (int, float, np.ndarray, Variable))

        opt = cal_node(mul_node_float32())
        res = Variable()

        if isinstance(other, (int, float, np.ndarray)):
            tem = Variable(other)

            #搭建数据流图
            tem.py_node.AddBackNode(opt)
            self.py_node.AddBackNode(opt)
            opt.AddPreNode(self.py_node)
            opt.AddPreNode(tem.py_node)

            opt.SetBackNode(res.py_node)
            res.py_node.SetPreNode(opt)

            #搭建节点和数据流图的关系
            if self.py_node.node.my_map != None:
                self.py_node.node.my_map.AddBeginNode(tem.py_node.node)
                self.py_node.node.my_map.AddDataNode(tem.py_node.node)
                self.py_node.node.my_map.AddDataNode(res.py_node.node)

                tem.py_node.node.SetMyMap(self.py_node.node.my_map)
                res.py_node.node.SetMyMap(self.py_node.node.my_map)

            return res
        else:
            m1, m2 = self.py_node.node.my_map, other.py_node.node.my_map
            if m1 == None and m2 != None:
                #搭建节点和数据流图的关系
                other.py_node.node.my_map.AddBeginNode(self.py_node.node)
                other.py_node.node.my_map.AddDataNode(self.py_node.node)
                other.py_node.node.my_map.AddDataNode(res.py_node.node)

                self.py_node.node.SetMyMap(other.py_node.node.my_map)
                res.py_node.node.SetMyMap(other.py_node.node.my_map)

                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(self.py_node)
                opt.AddPreNode(other.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            elif m1 != None and m2 == None:
                #搭建节点和数据流图的关系
                self.py_node.node.my_map.AddBeginNode(other.py_node.node)
                self.py_node.node.my_map.AddDataNode(other.py_node.node)
                self.py_node.node.my_map.AddDataNode(res.py_node.node)

                other.py_node.node.SetMyMap(self.py_node.node.my_map)
                res.py_node.node.SetMyMap(self.py_node.node.my_map)

                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(self.py_node)
                opt.AddPreNode(other.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            elif m1 == None and m2 == None:
                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(self.py_node)
                opt.AddPreNode(other.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            else:
                if self.py_node.node.my_map == other.py_node.node.my_map:
                    #搭建节点和数据流图关系
                    self.py_node.node.my_map.AddDataNode(res.py_node.node)
                    res.py_node.node.SetMyMap(self.py_node.node.my_map)

                    #搭建数据流图
                    other.py_node.AddBackNode(opt)
                    self.py_node.AddBackNode(opt)
                    opt.AddPreNode(self.py_node)
                    opt.AddPreNode(other.py_node)

                    opt.SetBackNode(res.py_node)
                    res.py_node.SetPreNode(opt)
                else:
                    print('节点错误：暂时不支持多数据流图混合编程！！')

            return res

    def __rmul__(self, other):
        assert isinstance(other, (int, float, np.ndarray, Variable))

        opt = cal_node(mul_node_float32())
        res = Variable()

        if isinstance(other, (int, float, np.ndarray)):
            tem = Variable(other)

            #搭建数据流图
            tem.py_node.AddBackNode(opt)
            self.py_node.AddBackNode(opt)
            opt.AddPreNode(tem.py_node)
            opt.AddPreNode(self.py_node)

            opt.SetBackNode(res.py_node)
            res.py_node.SetPreNode(opt)

            #搭建节点和数据流图的关系
            if self.py_node.node.my_map != None:
                self.py_node.node.my_map.AddBeginNode(tem.py_node.node)
                self.py_node.node.my_map.AddDataNode(tem.py_node.node)
                self.py_node.node.my_map.AddDataNode(res.py_node.node)

                tem.py_node.node.SetMyMap(self.py_node.node.my_map)
                res.py_node.node.SetMyMap(self.py_node.node.my_map)

            return res
        else:
            m1, m2 = self.py_node.node.my_map, other.py_node.node.my_map
            if m1 == None and m2 != None:
                #搭建节点和数据流图的关系
                other.py_node.node.my_map.AddBeginNode(self.py_node.node)
                other.py_node.node.my_map.AddDataNode(self.py_node.node)
                other.py_node.node.my_map.AddDataNode(res.py_node.node)

                self.py_node.node.SetMyMap(other.py_node.node.my_map)
                res.py_node.node.SetMyMap(other.py_node.node.my_map)

                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(other.py_node)
                opt.AddPreNode(self.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            elif m1 != None and m2 == None:
                #搭建节点和数据流图的关系
                self.py_node.node.my_map.AddBeginNode(other.py_node.node)
                self.py_node.node.my_map.AddDataNode(other.py_node.node)
                self.py_node.node.my_map.AddDataNode(res.py_node.node)

                other.py_node.node.SetMyMap(self.py_node.node.my_map)
                res.py_node.node.SetMyMap(self.py_node.node.my_map)

                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(other.py_node)
                opt.AddPreNode(self.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            elif m1 == None and m2 == None:
                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(other.py_node)
                opt.AddPreNode(self.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            else:
                if self.py_node.node.my_map == other.py_node.node.my_map:
                    #搭建节点和数据流图关系
                    self.py_node.node.my_map.AddDataNode(res.py_node.node)
                    res.py_node.node.SetMyMap(self.py_node.node.my_map)

                    #搭建数据流图
                    other.py_node.AddBackNode(opt)
                    self.py_node.AddBackNode(opt)
                    opt.AddPreNode(other.py_node)
                    opt.AddPreNode(self.py_node)

                    opt.SetBackNode(res.py_node)
                    res.py_node.SetPreNode(opt)
                else:
                    print('节点错误：暂时不支持多数据流图混合编程！！')

            return res

    def __truediv__(self, other):
        assert isinstance(other, (int, float, np.ndarray, Variable))

        opt = cal_node(div_node_float32())
        res = Variable()

        if isinstance(other, (int, float, np.ndarray)):
            tem = Variable(other)

            #搭建数据流图
            tem.py_node.AddBackNode(opt)
            self.py_node.AddBackNode(opt)
            opt.AddPreNode(self.py_node)
            opt.AddPreNode(tem.py_node)

            opt.SetBackNode(res.py_node)
            res.py_node.SetPreNode(opt)

            #搭建节点和数据流图的关系
            if self.py_node.node.my_map != None:
                self.py_node.node.my_map.AddBeginNode(tem.py_node.node)
                self.py_node.node.my_map.AddDataNode(tem.py_node.node)
                self.py_node.node.my_map.AddDataNode(res.py_node.node)

                tem.py_node.node.SetMyMap(self.py_node.node.my_map)
                res.py_node.node.SetMyMap(self.py_node.node.my_map)

            return res
        else:
            m1, m2 = self.py_node.node.my_map, other.py_node.node.my_map
            if m1 == None and m2 != None:
                #搭建节点和数据流图的关系
                other.py_node.node.my_map.AddBeginNode(self.py_node.node)
                other.py_node.node.my_map.AddDataNode(self.py_node.node)
                other.py_node.node.my_map.AddDataNode(res.py_node.node)

                self.py_node.node.SetMyMap(other.py_node.node.my_map)
                res.py_node.node.SetMyMap(other.py_node.node.my_map)

                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(self.py_node)
                opt.AddPreNode(other.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            elif m1 != None and m2 == None:
                #搭建节点和数据流图的关系
                self.py_node.node.my_map.AddBeginNode(other.py_node.node)
                self.py_node.node.my_map.AddDataNode(other.py_node.node)
                self.py_node.node.my_map.AddDataNode(res.py_node.node)

                other.py_node.node.SetMyMap(self.py_node.node.my_map)
                res.py_node.node.SetMyMap(self.py_node.node.my_map)

                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(self.py_node)
                opt.AddPreNode(other.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            elif m1 == None and m2 == None:
                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(self.py_node)
                opt.AddPreNode(other.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            else:
                if self.py_node.node.my_map == other.py_node.node.my_map:
                    #搭建节点和数据流图关系
                    self.py_node.node.my_map.AddDataNode(res.py_node.node)
                    res.py_node.node.SetMyMap(self.py_node.node.my_map)

                    #搭建数据流图
                    other.py_node.AddBackNode(opt)
                    self.py_node.AddBackNode(opt)
                    opt.AddPreNode(self.py_node)
                    opt.AddPreNode(other.py_node)

                    opt.SetBackNode(res.py_node)
                    res.py_node.SetPreNode(opt)
                else:
                    print('节点错误：暂时不支持多数据流图混合编程！！')

            return res

    def __rtruediv__(self, other):
        assert isinstance(other, (int, float, np.ndarray, Variable))

        opt = cal_node(div_node_float32())
        res = Variable()

        if isinstance(other, (int, float, np.ndarray)):
            tem = Variable(other)

            #搭建数据流图
            tem.py_node.AddBackNode(opt)
            self.py_node.AddBackNode(opt)
            opt.AddPreNode(tem.py_node)
            opt.AddPreNode(self.py_node)

            opt.SetBackNode(res.py_node)
            res.py_node.SetPreNode(opt)

            #搭建节点和数据流图的关系
            if self.py_node.node.my_map != None:
                self.py_node.node.my_map.AddBeginNode(tem.py_node.node)
                self.py_node.node.my_map.AddDataNode(tem.py_node.node)
                self.py_node.node.my_map.AddDataNode(res.py_node.node)

                tem.py_node.node.SetMyMap(self.py_node.node.my_map)
                res.py_node.node.SetMyMap(self.py_node.node.my_map)

            return res
        else:
            m1, m2 = self.py_node.node.my_map, other.py_node.node.my_map
            if m1 == None and m2 != None:
                #搭建节点和数据流图的关系
                other.py_node.node.my_map.AddBeginNode(self.py_node.node)
                other.py_node.node.my_map.AddDataNode(self.py_node.node)
                other.py_node.node.my_map.AddDataNode(res.py_node.node)

                self.py_node.node.SetMyMap(other.py_node.node.my_map)
                res.py_node.node.SetMyMap(other.py_node.node.my_map)

                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(other.py_node)
                opt.AddPreNode(self.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            elif m1 != None and m2 == None:
                #搭建节点和数据流图的关系
                self.py_node.node.my_map.AddBeginNode(other.py_node.node)
                self.py_node.node.my_map.AddDataNode(other.py_node.node)
                self.py_node.node.my_map.AddDataNode(res.py_node.node)

                other.py_node.node.SetMyMap(self.py_node.node.my_map)
                res.py_node.node.SetMyMap(self.py_node.node.my_map)

                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(other.py_node)
                opt.AddPreNode(self.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            elif m1 == None and m2 == None:
                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(other.py_node)
                opt.AddPreNode(self.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            else:
                if self.py_node.node.my_map == other.py_node.node.my_map:
                    #搭建节点和数据流图关系
                    self.py_node.node.my_map.AddDataNode(res.py_node.node)
                    res.py_node.node.SetMyMap(self.py_node.node.my_map)

                    #搭建数据流图
                    other.py_node.AddBackNode(opt)
                    self.py_node.AddBackNode(opt)
                    opt.AddPreNode(other.py_node)
                    opt.AddPreNode(self.py_node)

                    opt.SetBackNode(res.py_node)
                    res.py_node.SetPreNode(opt)
                else:
                    print('节点错误：暂时不支持多数据流图混合编程！！')

            return res

    def __pow__(self, other):
        assert isinstance(other, (int, float, np.ndarray, Variable))

        opt = cal_node(power_node_float32())
        res = Variable()

        if isinstance(other, (int, float, np.ndarray)):
            tem = Variable(other)

            #搭建数据流图
            tem.py_node.AddBackNode(opt)
            self.py_node.AddBackNode(opt)
            opt.AddPreNode(self.py_node)
            opt.AddPreNode(tem.py_node)

            opt.SetBackNode(res.py_node)
            res.py_node.SetPreNode(opt)

            #搭建节点和数据流图的关系
            if self.py_node.node.my_map != None:
                self.py_node.node.my_map.AddBeginNode(tem.py_node.node)
                self.py_node.node.my_map.AddDataNode(tem.py_node.node)
                self.py_node.node.my_map.AddDataNode(res.py_node.node)

                tem.py_node.node.SetMyMap(self.py_node.node.my_map)
                res.py_node.node.SetMyMap(self.py_node.node.my_map)

            return res
        else:
            m1, m2 = self.py_node.node.my_map, other.py_node.node.my_map
            if m1 == None and m2 != None:
                #搭建节点和数据流图的关系
                other.py_node.node.my_map.AddBeginNode(self.py_node.node)
                other.py_node.node.my_map.AddDataNode(self.py_node.node)
                other.py_node.node.my_map.AddDataNode(res.py_node.node)

                self.py_node.node.SetMyMap(other.py_node.node.my_map)
                res.py_node.node.SetMyMap(other.py_node.node.my_map)

                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(self.py_node)
                opt.AddPreNode(other.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            elif m1 != None and m2 == None:
                #搭建节点和数据流图的关系
                self.py_node.node.my_map.AddBeginNode(other.py_node.node)
                self.py_node.node.my_map.AddDataNode(other.py_node.node)
                self.py_node.node.my_map.AddDataNode(res.py_node.node)

                other.py_node.node.SetMyMap(self.py_node.node.my_map)
                res.py_node.node.SetMyMap(self.py_node.node.my_map)

                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(self.py_node)
                opt.AddPreNode(other.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            elif m1 == None and m2 == None:
                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(self.py_node)
                opt.AddPreNode(other.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            else:
                if self.py_node.node.my_map == other.py_node.node.my_map:
                    #搭建节点和数据流图关系
                    self.py_node.node.my_map.AddDataNode(res.py_node.node)
                    res.py_node.node.SetMyMap(self.py_node.node.my_map)

                    #搭建数据流图
                    other.py_node.AddBackNode(opt)
                    self.py_node.AddBackNode(opt)
                    opt.AddPreNode(self.py_node)
                    opt.AddPreNode(other.py_node)

                    opt.SetBackNode(res.py_node)
                    res.py_node.SetPreNode(opt)
                else:
                    print('节点错误：暂时不支持多数据流图混合编程！！')

            return res

    def __rpow__(self, other):
        assert isinstance(other, (int, float, np.ndarray, Variable))

        opt = cal_node(power_node_float32())
        res = Variable()

        if isinstance(other, (int, float, np.ndarray)):
            tem = Variable(other)

            #搭建数据流图
            tem.py_node.AddBackNode(opt)
            self.py_node.AddBackNode(opt)
            opt.AddPreNode(tem.py_node)
            opt.AddPreNode(self.py_node)

            opt.SetBackNode(res.py_node)
            res.py_node.SetPreNode(opt)

            #搭建节点和数据流图的关系
            if self.py_node.node.my_map != None:
                self.py_node.node.my_map.AddBeginNode(tem.py_node.node)
                self.py_node.node.my_map.AddDataNode(tem.py_node.node)
                self.py_node.node.my_map.AddDataNode(res.py_node.node)

                tem.py_node.node.SetMyMap(self.py_node.node.my_map)
                res.py_node.node.SetMyMap(self.py_node.node.my_map)

            return res
        else:
            m1, m2 = self.py_node.node.my_map, other.py_node.node.my_map
            if m1 == None and m2 != None:
                #搭建节点和数据流图的关系
                other.py_node.node.my_map.AddBeginNode(self.py_node.node)
                other.py_node.node.my_map.AddDataNode(self.py_node.node)
                other.py_node.node.my_map.AddDataNode(res.py_node.node)

                self.py_node.node.SetMyMap(other.py_node.node.my_map)
                res.py_node.node.SetMyMap(other.py_node.node.my_map)

                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(other.py_node)
                opt.AddPreNode(self.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            elif m1 != None and m2 == None:
                #搭建节点和数据流图的关系
                self.py_node.node.my_map.AddBeginNode(other.py_node.node)
                self.py_node.node.my_map.AddDataNode(other.py_node.node)
                self.py_node.node.my_map.AddDataNode(res.py_node.node)

                other.py_node.node.SetMyMap(self.py_node.node.my_map)
                res.py_node.node.SetMyMap(self.py_node.node.my_map)

                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(other.py_node)
                opt.AddPreNode(self.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            elif m1 == None and m2 == None:
                #搭建数据流图
                other.py_node.AddBackNode(opt)
                self.py_node.AddBackNode(opt)
                opt.AddPreNode(other.py_node)
                opt.AddPreNode(self.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)

            else:
                if self.py_node.node.my_map == other.py_node.node.my_map:
                    #搭建节点和数据流图关系
                    self.py_node.node.my_map.AddDataNode(res.py_node.node)
                    res.py_node.node.SetMyMap(self.py_node.node.my_map)

                    #搭建数据流图
                    other.py_node.AddBackNode(opt)
                    self.py_node.AddBackNode(opt)
                    opt.AddPreNode(other.py_node)
                    opt.AddPreNode(self.py_node)

                    opt.SetBackNode(res.py_node)
                    res.py_node.SetPreNode(opt)
                else:
                    print('节点错误：暂时不支持多数据流图混合编程！！')

            return res


# m = map()

# x = Variable(2)

# x >> m

# res = 3 * x**2 + 2 * x + 1

# m.Forward()

# useless = 1

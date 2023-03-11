import sys

sys.path.append('.')
from Pypanda.dyc_lib.data_node_float32 import data_node_float32, const_node_float32
from Pypanda.dyc_lib.opt_node_float32 import add_node_float32, mul_node_float32, sub_node_float32, div_node_float32, power_node_float32, exp_node_float32, ln_node_float32, sin_node_float32, cos_node_float32, tan_node_float32
from Pypanda.dyc_lib.cal_map_flow_float32 import cal_map_float32
from Pypanda.dyc_lib.loss_func_float32 import mseloss_node_float32,celoss_node_float32
from Pypanda.variable import *


def MSELoss(node1, node2):
    assert isinstance(node1, (int, float, np.ndarray, Variable))
    assert isinstance(node2, (int, float, np.ndarray, Variable))

    res = Variable()
    opt = cal_node(mseloss_node_float32())

    if isinstance(node1,
                  (int, float, np.ndarray)) and isinstance(node2, (Variable)):
        tem = Variable(node1)

        #搭建数据流图
        tem.py_node.AddBackNode(opt)
        node2.py_node.AddBackNode(opt)
        opt.AddPreNode(node2.py_node)
        opt.AddPreNode(tem.py_node)

        opt.SetBackNode(res.py_node)
        res.py_node.SetPreNode(opt)

        #搭建节点和数据流图的关系
        if node2.py_node.node.my_map != None:
            node2.py_node.node.my_map.AddBeginNode(tem.py_node.node)
            node2.py_node.node.my_map.AddDataNode(tem.py_node.node)

            #默认如果有损失函数即当作end节点
            # node2.py_node.node.my_map.AddEndNode(res.py_node.node)
            node2.py_node.node.my_map.AddDataNode(res.py_node.node)

            tem.py_node.node.SetMyMap(node2.py_node.node.my_map)
            res.py_node.node.SetMyMap(node2.py_node.node.my_map)

    elif isinstance(node2, (int, float, np.ndarray)) and isinstance(
            node1, (Variable)):

        tem = Variable(node2)

        #搭建数据流图
        tem.py_node.AddBackNode(opt)
        node1.py_node.AddBackNode(opt)
        opt.AddPreNode(node1.py_node)
        opt.AddPreNode(tem.py_node)

        opt.SetBackNode(res.py_node)
        res.py_node.SetPreNode(opt)

        #搭建节点和数据流图的关系
        if node1.py_node.node.my_map != None:
            node1.py_node.node.my_map.AddBeginNode(tem.py_node.node)
            node1.py_node.node.my_map.AddDataNode(tem.py_node.node)

            #默认如果有损失函数即当作end节点
            # node1.py_node.node.my_map.AddEndNode(res.py_node.node)
            node1.py_node.node.my_map.AddDataNode(res.py_node.node)

            tem.py_node.node.SetMyMap(node1.py_node.node.my_map)
            res.py_node.node.SetMyMap(node1.py_node.node.my_map)

    elif isinstance(node2, (int, float, np.ndarray)) and isinstance(
            node1, (int, float, np.ndarray)):
        print('该函数主要用于构建数据流图，参数中至少需要一个Variable变量')

    else:
        m1, m2 = node1.py_node.node.my_map, node2.py_node.node.my_map
        if m1 == None and m2 != None:
            #搭建节点和数据流图的关系
            node2.py_node.node.my_map.AddBeginNode(node1.py_node.node)
            node2.py_node.node.my_map.AddDataNode(node1.py_node.node)

            # node2.py_node.node.my_map.AddEndNode(res.py_node.node)
            node2.py_node.node.my_map.AddDataNode(res.py_node.node)

            node1.py_node.node.SetMyMap(node2.py_node.node.my_map)
            res.py_node.node.SetMyMap(node2.py_node.node.my_map)

            #搭建数据流图
            node2.py_node.AddBackNode(opt)
            node1.py_node.AddBackNode(opt)
            opt.AddPreNode(node1.py_node)
            opt.AddPreNode(node2.py_node)

            opt.SetBackNode(res.py_node)
            res.py_node.SetPreNode(opt)

        elif m1 != None and m2 == None:
            #搭建节点和数据流图的关系
            node1.py_node.node.my_map.AddBeginNode(node2.py_node.node)
            node1.py_node.node.my_map.AddDataNode(node2.py_node.node)

            # node1.py_node.node.my_map.AddEndNode(res.py_node.node)
            node1.py_node.node.my_map.AddDataNode(res.py_node.node)

            node2.py_node.node.SetMyMap(node1.py_node.node.my_map)
            res.py_node.node.SetMyMap(node1.py_node.node.my_map)

            #搭建数据流图
            node2.py_node.AddBackNode(opt)
            node1.py_node.AddBackNode(opt)
            opt.AddPreNode(node1.py_node)
            opt.AddPreNode(node2.py_node)

            opt.SetBackNode(res.py_node)
            res.py_node.SetPreNode(opt)

        elif m1 == None and m2 == None:
            #搭建数据流图
            node2.py_node.AddBackNode(opt)
            node1.py_node.AddBackNode(opt)
            opt.AddPreNode(node1.py_node)
            opt.AddPreNode(node2.py_node)

            opt.SetBackNode(res.py_node)
            res.py_node.SetPreNode(opt)

        else:
            if node1.py_node.node.my_map == node2.py_node.node.my_map:
                #搭建节点和数据流图关系
                node1.py_node.node.my_map.AddDataNode(res.py_node.node)
                # node1.py_node.node.my_map.AddEndNode(res.py_node.node)
                res.py_node.node.SetMyMap(node1.py_node.node.my_map)

                #搭建数据流图
                node2.py_node.AddBackNode(opt)
                node1.py_node.AddBackNode(opt)
                opt.AddPreNode(node1.py_node)
                opt.AddPreNode(node2.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)
            else:
                print('节点错误：暂时不支持多数据流图混合编程！！')

    return res

def CELoss(node1, node2):
    assert isinstance(node1, (int, float, np.ndarray, Variable))
    assert isinstance(node2, (int, float, np.ndarray, Variable))

    res = Variable()
    opt = cal_node(celoss_node_float32())

    if isinstance(node1,
                  (int, float, np.ndarray)) and isinstance(node2, (Variable)):
        tem = Variable(node1)

        #搭建数据流图
        tem.py_node.AddBackNode(opt)
        node2.py_node.AddBackNode(opt)
        opt.AddPreNode(node2.py_node)
        opt.AddPreNode(tem.py_node)

        opt.SetBackNode(res.py_node)
        res.py_node.SetPreNode(opt)

        #搭建节点和数据流图的关系
        if node2.py_node.node.my_map != None:
            node2.py_node.node.my_map.AddBeginNode(tem.py_node.node)
            node2.py_node.node.my_map.AddDataNode(tem.py_node.node)

            #默认如果有损失函数即当作end节点
            # node2.py_node.node.my_map.AddEndNode(res.py_node.node)
            node2.py_node.node.my_map.AddDataNode(res.py_node.node)

            tem.py_node.node.SetMyMap(node2.py_node.node.my_map)
            res.py_node.node.SetMyMap(node2.py_node.node.my_map)

    elif isinstance(node2, (int, float, np.ndarray)) and isinstance(
            node1, (Variable)):

        tem = Variable(node2)

        #搭建数据流图
        tem.py_node.AddBackNode(opt)
        node1.py_node.AddBackNode(opt)
        opt.AddPreNode(node1.py_node)
        opt.AddPreNode(tem.py_node)

        opt.SetBackNode(res.py_node)
        res.py_node.SetPreNode(opt)

        #搭建节点和数据流图的关系
        if node1.py_node.node.my_map != None:
            node1.py_node.node.my_map.AddBeginNode(tem.py_node.node)
            node1.py_node.node.my_map.AddDataNode(tem.py_node.node)

            #默认如果有损失函数即当作end节点
            # node1.py_node.node.my_map.AddEndNode(res.py_node.node)
            node1.py_node.node.my_map.AddDataNode(res.py_node.node)

            tem.py_node.node.SetMyMap(node1.py_node.node.my_map)
            res.py_node.node.SetMyMap(node1.py_node.node.my_map)

    elif isinstance(node2, (int, float, np.ndarray)) and isinstance(
            node1, (int, float, np.ndarray)):
        print('该函数主要用于构建数据流图，参数中至少需要一个Variable变量')

    else:
        m1, m2 = node1.py_node.node.my_map, node2.py_node.node.my_map
        if m1 == None and m2 != None:
            #搭建节点和数据流图的关系
            node2.py_node.node.my_map.AddBeginNode(node1.py_node.node)
            node2.py_node.node.my_map.AddDataNode(node1.py_node.node)

            # node2.py_node.node.my_map.AddEndNode(res.py_node.node)
            node2.py_node.node.my_map.AddDataNode(res.py_node.node)

            node1.py_node.node.SetMyMap(node2.py_node.node.my_map)
            res.py_node.node.SetMyMap(node2.py_node.node.my_map)

            #搭建数据流图
            node2.py_node.AddBackNode(opt)
            node1.py_node.AddBackNode(opt)
            opt.AddPreNode(node1.py_node)
            opt.AddPreNode(node2.py_node)

            opt.SetBackNode(res.py_node)
            res.py_node.SetPreNode(opt)

        elif m1 != None and m2 == None:
            #搭建节点和数据流图的关系
            node1.py_node.node.my_map.AddBeginNode(node2.py_node.node)
            node1.py_node.node.my_map.AddDataNode(node2.py_node.node)

            # node1.py_node.node.my_map.AddEndNode(res.py_node.node)
            node1.py_node.node.my_map.AddDataNode(res.py_node.node)

            node2.py_node.node.SetMyMap(node1.py_node.node.my_map)
            res.py_node.node.SetMyMap(node1.py_node.node.my_map)

            #搭建数据流图
            node2.py_node.AddBackNode(opt)
            node1.py_node.AddBackNode(opt)
            opt.AddPreNode(node1.py_node)
            opt.AddPreNode(node2.py_node)

            opt.SetBackNode(res.py_node)
            res.py_node.SetPreNode(opt)

        elif m1 == None and m2 == None:
            #搭建数据流图
            node2.py_node.AddBackNode(opt)
            node1.py_node.AddBackNode(opt)
            opt.AddPreNode(node1.py_node)
            opt.AddPreNode(node2.py_node)

            opt.SetBackNode(res.py_node)
            res.py_node.SetPreNode(opt)

        else:
            if node1.py_node.node.my_map == node2.py_node.node.my_map:
                #搭建节点和数据流图关系
                node1.py_node.node.my_map.AddDataNode(res.py_node.node)
                # node1.py_node.node.my_map.AddEndNode(res.py_node.node)
                res.py_node.node.SetMyMap(node1.py_node.node.my_map)

                #搭建数据流图
                node2.py_node.AddBackNode(opt)
                node1.py_node.AddBackNode(opt)
                opt.AddPreNode(node1.py_node)
                opt.AddPreNode(node2.py_node)

                opt.SetBackNode(res.py_node)
                res.py_node.SetPreNode(opt)
            else:
                print('节点错误：暂时不支持多数据流图混合编程！！')

    return res

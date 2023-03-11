#ifndef NODE
#define NODE
#include <vector>
#include "../../lib/xtensor/include/xtensor/xarray.hpp"
// #include "../utils/cal_map_flow.hpp"
#include "../datatype.h"
#include <string>
#include <set>
 #include "../../lib/pybind11/include/pybind11/pybind11.h"

#define DATA_NODE_FLOAT32 __DATA_NODE__<FLOAT32>
#define CONST_NODE_FLOAT32 __CONST_NODE__<FLOAT32>
#define CAL_NODE_FLOAT32 __CAL_NODE__<FLOAT32>
using namespace xt;

// class __NODE__
// {

// public:
//     std::vector<__NODE__ *> pre_node;
//     std::vector<__NODE__ *> back_node;
//     bool require_grad;
// };

template <typename T>
class __CAL_MAP__;

template <typename T>
class __CAL_NODE__;

template <typename T>
class __DATA_NODE__
{
public:
    __CAL_NODE__<T> *pre_node;
    std::vector<__CAL_NODE__<T> *> back_node;
    xarray<T> data;
    xarray<T> grad;
    std::set<__CAL_NODE__<T> *> if_backward;
    bool require_grad;
    __CAL_MAP__<T> *my_map;
    std::string my_properity = "DATA_NODE";

    __DATA_NODE__()
    {
        this->pre_node = nullptr;
        this->my_map = nullptr;
        this->require_grad = true;
        this->data = xarray<T>({0});
        this->grad = xarray<T>({0});
    }
    __DATA_NODE__(xarray<T> &data)
    {
        this->pre_node = nullptr;
        this->my_map = nullptr;
        this->require_grad = true;
        this->data = data;
        this->grad = xarray<T>({0});
    }

    __DATA_NODE__(const xarray<T> &data)
    {
        this->pre_node = nullptr;
        this->my_map = nullptr;
        this->require_grad = true;
        this->data = data;
        this->grad = xarray<T>({0});
    }

    void SetPreNode(__CAL_NODE__<T> *pre_node)
    {
        this->pre_node = pre_node;
    }

    void SetBackNode(std::vector<__CAL_NODE__<T> *> &back_node)
    {
        this->back_node = back_node;
    }

    void SetBackNode(const std::vector<__CAL_NODE__<T> *> &back_node)
    {
        this->back_node = back_node;
    }

    void AddBackNode(__CAL_NODE__<T> *back_node)
    {
        this->back_node.push_back(back_node);
        // this->if_backward.push_back(false);
    }

    void SetData(xarray<T> &data)
    {
        this->data = data;
    }

    void SetData(const xarray<T> &data)
    {
        this->data = data;
    }

    void SetGrad(xarray<T> &grad)
    {
        this->grad = grad;
    }

    void SetGrad(const xarray<T> &grad)
    {
        this->grad = grad;
    }

    void Backward()
    {
        this->if_backward.clear();
    }

    void SetMyMap(__CAL_MAP__<T> *my_map)
    {
        this->my_map = my_map;
    }

    // void doit()
    // {
    //     this->pre_node->Forward();
    // }
};

template <typename T>
class __CONST_NODE__ : public __DATA_NODE__<T>
{
public:
    __CONST_NODE__()
    {
        this->pre_node = nullptr;
        this->my_map = nullptr;
        this->require_grad = false;
        this->data = xarray<T>({0});
        this->grad = xarray<T>({0});
    }

    __CONST_NODE__(xarray<T> &data)
    {
        this->pre_node = nullptr;
        this->my_map = nullptr;
        this->require_grad = false;
        this->data = data;
        this->grad = xarray<T>({0});
    }

    __CONST_NODE__(const xarray<T> &data)
    {
        this->pre_node = nullptr;
        this->my_map = nullptr;
        this->require_grad = true;
        this->data = data;
        this->grad = xarray<T>({0});
    }
};

template <typename T>
class __CAL_NODE__
{
public:
    std::vector<__DATA_NODE__<T> *> pre_node;
    __DATA_NODE__<T> *back_node;
    std::set<__DATA_NODE__<T> *> if_forward;
    std::string my_properity = "CAL_NODE";

    __CAL_NODE__()
    {
        this->back_node = nullptr;
    }

    void SetPreNode(std::vector<__DATA_NODE__<T> *> &pre_node)
    {
        this->pre_node = pre_node;
    }

    void SetBackNode(__DATA_NODE__<T> *back_node)
    {
        this->back_node = back_node;
    }

    void AddPreNode(__DATA_NODE__<T> *pre_node)
    {
        this->pre_node.push_back(pre_node);
    }

    virtual void Forward() = 0;
    virtual void Backward() = 0;
    // virtual void ZeroGrad() { std::cout << "调用的是我，绷不住了！！"; }
    // virtual void UpdatePara(T eta) { std::cout << "调用的还是我，绷不住了！！"; }
    virtual void ZeroGrad(){}
    virtual void UpdatePara(T eta){}
};

 class Py_cal_node_float32 : public CAL_NODE_FLOAT32
 {
 public:
     using __CAL_NODE__::__CAL_NODE__;

     void Forward() override
     {
         PYBIND11_OVERLOAD_PURE(
             void,
             __CAL_NODE__<FLOAT32>,
             Forward);
     }

     void Backward() override
     {
         PYBIND11_OVERLOAD_PURE(
             void,
             __CAL_NODE__<FLOAT32>,
             Backward);
     }
 };

#endif
#ifndef RESHAPE
#define RESHAPE

#include <ctime>
#include "../datatype.h"
#include "../node/node.hpp"
#include "../utils/matmul.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xslice.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xadapt.hpp>
#include<xtensor/xlayout.hpp>
#include <exception>
#include <vector>
#include <queue>
#include <string>
#include <math.h>

using namespace xt;

#define RESHAPE_NODE_FLOAT32 __RESHAPE_NODE__<FLOAT32>

template <typename T>
class __RESHAPE_NODE__ : public __CAL_NODE__<T>
{
public:
    std::vector<INT32> output_shape;

    __RESHAPE_NODE__(std::vector<INT32> shape)
    {
        this->output_shape = shape;
    }

    void AddPreNode(__DATA_NODE__<T>* pre_node)
    {
        try
        {
            if (this->pre_node.size() >= 1)
                throw "The \"pre_node\" of \"__RESHAPE_NODE__\" has been overflowed!!";
        }
        catch (const char* s)
        {
            std::cout << s << '\n';
        }

        this->pre_node.push_back(pre_node);
    }

    void Forward()
    {
        try
        {
            if (this->pre_node.size() < 1)
                throw "The value of length of the \"__RESHAPE_NODE__\" need to be 2,but the actual value of it is not enough!";
        }
        catch (const char* s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"back_node\" of \"__RESHAPE_NODE__\" is nullptr!";
        }
        catch (const char* s)
        {
            std::cout << s << '\n';
        }

        try
        {
            // 正向传播
            auto tem = this->pre_node[0]->data;
            this->back_node->data = tem.reshape(this->output_shape);
        }
        catch (std::exception& e)
        {
            std::cout << "__RESHAPE_NODE__Forward:" << e.what() << std::endl;
        }
    }

    void Backward()
    {
        try
        {
            // 反向传播
            auto input_shape = this->pre_node[0]->data.shape();
            auto tem = this->back_node->grad;
            this->pre_node[0]->grad += tem.reshape(input_shape);
        }
        catch (std::exception& e)
        {
            std::cout << "__RESHAPE_NODE__Backward:" << e.what() << std::endl;
        }
    }
};
#endif
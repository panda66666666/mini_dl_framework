#ifndef SIGMOID
#define SIGMOID
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
#include <exception>
#include <vector>
#include <queue>
#include <string>
#include <math.h>

using namespace xt;

#define SIGMOID_NODE_FLOAT32 __SIGMOID_NODE__<FLOAT32>

template <typename T>
class __SIGMOID_NODE__ : public __CAL_NODE__<T>
{
public:
    void AddPreNode(__DATA_NODE__<T> *pre_node)
    {
        try
        {
            if (this->pre_node.size() >= 1)
                throw "The \"pre_node\" of \"__SIGMOID_NODE__\" has been overflowed!!";
        }
        catch (const char *s)
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
                throw "The value of length of the \"__SIGMOID_NODE__\" need to be 2,but the actual value of it is not enough!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"back_node\" of \"__SIGMOID_NODE__\" is nullptr!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            // 正向传播
            this->back_node->data = 1 / (1 + exp(-this->pre_node[0]->data));
            this->if_forward.clear();
        }
        catch (std::exception &e)
        {
            std::cout << "__SIGMOID_NODE__Forward:" << e.what() << std::endl;
        }
    }

    void Backward()
    {
        try
        {
            // 反向传播
            this->pre_node[0]->grad += this->back_node->grad * this->back_node->data * (1 - this->back_node->data);
            auto useless = 0;
        }
        catch (std::exception &e)
        {
            std::cout << "__SIGMOID_NODE__Backward:" << e.what() << std::endl;
        }
    }
};

#endif
#ifndef SOFTMAX
#define SOFTMAX
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

#define SOFTMAX_NODE_FLOAT32 __SOFTMAX_NODE__<FLOAT32>

template <typename T>
class __SOFTMAX_NODE__ : public __CAL_NODE__<T>
{
public:
    void AddPreNode(__DATA_NODE__<T> *pre_node)
    {
        try
        {
            if (this->pre_node.size() >= 1)
                throw "The \"pre_node\" of \"__SOFTMAX_NODE__\" has been overflowed!!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        this->pre_node.push_back(pre_node);
    }

    xarray<T> RecuritFW(const xarray<T> &input)
    {
        xarray<UNSIGNED_INT32>::shape_type shape = input.shape();
        xarray<T> out = xarray<T>(input.shape());

        if (shape.size() == 1)
        {
            out = exp(input) / sum(exp(input), 0);
        }
        else
        {
            for (UNSIGNED_INT32 i = 0; i < shape[0]; i++)
            {
                view(out, i, all()) = RecuritFW(view(input, i, all()));
            }
        }

        return out;
    }

    xarray<T> RecuritBW(const xarray<T> &input, const xarray<T> &output_grad)
    {
        xarray<UNSIGNED_INT32>::shape_type shape = input.shape();
        xarray<T> out = xarray<T>(input.shape());

        if (shape.size() == 1)
        {
            auto SUM = sum(exp(input), 0);

            // std::cout << exp(input) * ((SUM - exp(input)) / pow(SUM, 2) * output_grad) << std::endl;

            // std::cout << exp(input) * sum(exp(input) * output_grad / pow(SUM, 2), 0) << std::endl;

            // std::cout << pow(exp(input), 2) * output_grad / pow(SUM, 2) << std::endl;

            out = (exp(input) * (xarray<T>({1}) / SUM) * output_grad) -

                   (exp(input) * sum(exp(input) * output_grad / pow(SUM, 2), 0));
        }
        else
        {
            for (UNSIGNED_INT32 i = 0; i < shape[0]; i++)
            {
                view(out, i, all()) = RecuritBW(view(input, i, all()),
                                                view(output_grad, i, all()));
            }
        }

        return out;
    }

    void Forward()
    {
        try
        {
            if (this->pre_node.size() < 1)
                throw "The value of length of the \"__SOFTMAX_NODE__\" need to be 2,but the actual value of it is not enough!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"back_node\" of \"__SOFTMAX_NODE__\" is nullptr!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            // 正向传播
            this->back_node->data = this->RecuritFW(this->pre_node[0]->data);
        }
        catch (std::exception &e)
        {
            std::cout << "__SOFTMAX_NODE__Forward:" << e.what() << std::endl;
        }
    }

    void Backward()
    {
        try
        {
            // 反向传播
            this->pre_node[0]->grad += this->RecuritBW(this->pre_node[0]->data, this->back_node->grad);

            auto useless = 0;
        }
        catch (std::exception &e)
        {
            std::cout << "__SOFTMAX_NODE__Backward:" << e.what() << std::endl;
        }
    }
};

#endif
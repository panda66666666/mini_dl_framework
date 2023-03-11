#ifndef MSELOSS
#define MSELOSS

#include "../datatype.h"
#include "../node/node.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <exception>
#include <vector>
#include <queue>
#include <string>
using namespace xt;

#define MSELOSS_NODE_FLOAT32 __MSELOSS__<FLOAT32>

template <typename T>
class __MSELOSS__ : public __CAL_NODE__<T>
{
public:
    void AddPreNode(__DATA_NODE__<T> *pre_node)
    {
        try
        {
            if (this->pre_node.size() >= 2)
                throw "The \"pre_node\" of \"__MSE_NODE__\" has been overflowed!!";
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
            if (this->pre_node.size() < 2)
                throw "The value of length of the \"__MSE_NODE__\" need to be 2,but the actual value of it is not enough!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"baak_node\" of \"__MSE_NODE__\" is nullptr!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            this->back_node->data = mean(pow(this->pre_node[0]->data - this->pre_node[1]->data, 2));

            this->if_forward.clear();
        }
        catch (std::exception &e)
        {
            std::cout << "__MSE_NODE__Forward:" << e.what() << std::endl;
        }
    }

    void Backward()
    {
        try
        {
            auto tem = this->pre_node[0]->data;

            this->pre_node[0]->grad += (2 * tem - 2 * this->pre_node[1]->data) / tem.size();
            // std::cout << tem << this->pre_node[1]->data;

            this->pre_node[1]->grad += (2 * this->pre_node[1]->data - 2 * tem) / this->pre_node[1]->data.size();
        }
        catch (std::exception &e)
        {
            std::cout << "__MSE_NODE__Backward:" << e.what() << std::endl;
        }
    }
};

#endif
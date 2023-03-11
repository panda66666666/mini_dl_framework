#ifndef BASIC_OPERATION
#define BASIC_OPERATION
#include "../datatype.h"
#include "../node/node.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <exception>
#include <vector>
#include <string>
#include <eh.h>
using namespace xt;

#define ADD_NODE_FLOAT32 __ADD_NODE__<FLOAT32>
#define SUB_NODE_FLOAT32 __SUB_NODE__<FLOAT32>
#define MUL_NODE_FLOAT32 __MUL_NODE__<FLOAT32>
#define DIV_NODE_FLOAT32 __DIV_NODE__<FLOAT32>

template <typename T>
class __ADD_NODE__ : public __CAL_NODE__<T>
{
public:
    void AddPreNode(__DATA_NODE__<T> *pre_node)
    {
        try
        {
            if (this->pre_node.size() >= 2)
                throw "The \"pre_node\" of \"ADD_NODE\" has been overflowed!!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        this->pre_node.push_back(pre_node);
        // this->if_forward.push_back(false);
    }

    void Forward()
    {
        try
        {
            if (this->pre_node.size() < 2)
                throw "The value of length of the \"ADD_NODE\" need to be 2,but the actual value of it is not enough!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"baak_node\" of \"ADD_NODE\" is nullptr!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            this->back_node->data = this->pre_node[0]->data + this->pre_node[1]->data;

            this->if_forward.clear();
        }
        catch (std::exception &e)
        {
            std::cout << "__ADD_NODE__Forward:" << e.what() << std::endl;
        }

        // __CAL_NODE__<FLOAT32> *p_pet = new __ADD_NODE__<FLOAT32>();
        // __ADD_NODE__<FLOAT32> *p_dog = dynamic_cast<__ADD_NODE__<FLOAT32> *>(p_pet);
    }

    void Backward()
    {
        this->pre_node[0]->grad += this->back_node->grad;

        this->pre_node[1]->grad += this->back_node->grad;
    }

    // void bark()
    // {
    //     std::cout << "哈哈哈哈可以调用！！！";
    // }
};

template <typename T>
class __SUB_NODE__ : public __CAL_NODE__<T>
{
public:
    void AddPreNode(__DATA_NODE__<T> *pre_node)
    {
        try
        {
            if (this->pre_node.size() >= 2)
                throw "The \"pre_node\" of \"__SUB_NODE__\" has been overflowed!!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        this->pre_node.push_back(pre_node);
        // this->if_forward.push_back(false);
    }

    void Forward()
    {
        try
        {
            if (this->pre_node.size() < 2)
                throw "The value of length of the \"__SUB_NODE__\" need to be 2,but the actual value of it is not enough!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"baak_node\" of \"__SUB_NODE__\" is nullptr!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            this->back_node->data = this->pre_node[0]->data - this->pre_node[1]->data;

            this->if_forward.clear();
        }
        catch (std::exception &e)
        {
            std::cout << "__SUB_NODE__Forward:" << e.what() << std::endl;
        }
    }

    void Backward()
    {
        this->pre_node[0]->grad += this->back_node->grad;

        this->pre_node[1]->grad += -this->back_node->grad;
    }
};

template <typename T>
class __MUL_NODE__ : public __CAL_NODE__<T>
{
public:
    void AddPreNode(__DATA_NODE__<T> *pre_node)
    {
        try
        {
            if (this->pre_node.size() >= 2)
                throw "The \"pre_node\" of \"__MUL_NODE__\" has been overflowed!!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        this->pre_node.push_back(pre_node);
        // this->if_forward.push_back(false);
    }

    void Forward()
    {
        try
        {
            if (this->pre_node.size() < 2)
                throw "The value of length of the \"__MUL_NODE__\" need to be 2,but the actual value of it is not enough!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"baak_node\" of \"__MUL_NODE__\" is nullptr!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            this->back_node->data = this->pre_node[0]->data * this->pre_node[1]->data;

            this->if_forward.clear();
        }
        catch (std::exception &e)
        {
            std::cout << "__MUL_NODE__Forward:" << e.what() << std::endl;
        }
    }

    void Backward()
    {
        xarray<T> tem = this->pre_node[0]->data;

        this->pre_node[0]->grad += this->pre_node[1]->data * this->back_node->grad;

        this->pre_node[1]->grad += tem * this->back_node->grad;
    }
};

template <typename T>
class __DIV_NODE__ : public __CAL_NODE__<T>
{
public:
    void AddPreNode(__DATA_NODE__<T> *pre_node)
    {
        try
        {
            if (this->pre_node.size() >= 2)
                throw "The \"pre_node\" of \"__DIV_NODE__\" has been overflowed!!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        this->pre_node.push_back(pre_node);
        // this->if_forward.push_back(false);
    }

    void Forward()
    {
        try
        {
            if (this->pre_node.size() < 2)
                throw "The value of length of the \"__DIV_NODE__\" need to be 2,but the actual value of it is not enough!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"baak_node\" of \"__DIV_NODE__\" is nullptr!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            this->back_node->data = this->pre_node[0]->data / this->pre_node[1]->data;

            this->if_forward.clear();
        }
        catch (std::exception &e)
        {
            std::cout << "__DIV_NODE__Forward:" << e.what() << std::endl;
        }
    }

    void Backward()
    {
        /*
            a / b = c
            1/b
            -a/b^2
        */
        xarray<T> tem = this->pre_node[0]->data;

        this->pre_node[0]->grad += (ones<T>(this->pre_node[1]->data.shape()) / this->pre_node[1]->data) * this->back_node->grad;

        this->pre_node[1]->grad += -(tem * ((ones<T>(this->pre_node[1]->data.shape()) / this->pre_node[1]->data) * (ones<T>(this->pre_node[1]->data.shape()) / this->pre_node[1]->data)) * this->back_node->grad);
    }
};

#endif
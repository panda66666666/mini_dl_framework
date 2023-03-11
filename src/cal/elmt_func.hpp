#ifndef ELMT_FUNC
#define ELMT_FUNC
#include "../datatype.h"
#include "../node/node.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <exception>
#include <vector>
#include <string>
using namespace xt;

#define POWER_NODE_FLOAT32 __POWER_NODE__<FLOAT32>
#define EXP_NODE_FLOAT32 __EXP_NODE__<FLOAT32>
#define LN_NODE_FLOAT32 __LN_NODE__<FLOAT32>
#define SIN_NODE_FLOAT32 __SIN_NODE__<FLOAT32>
#define COS_NODE_FLOAT32 __COS_NODE__<FLOAT32>
#define TAN_NODE_FLOAT32 __TAN_NODE__<FLOAT32>

/*
    pre_node[0]^pre_node[1]
*/
template <typename T>
class __POWER_NODE__ : public __CAL_NODE__<T>
{
public:
    void AddPreNode(__DATA_NODE__<T> *pre_node)
    {
        try
        {
            if (this->pre_node.size() >= 2)
                throw "The \"pre_node\" of \"__POWER_NODE__\" has been overflowed!!";
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
                throw "The value of length of the \"__POWER_NODE__\" need to be 2,but the actual value of it is not enough!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"baak_node\" of \"__POWER_NODE__\" is nullptr!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            this->back_node->data = pow(this->pre_node[0]->data, this->pre_node[1]->data);

            this->if_forward.clear();
        }
        catch (std::exception &e)
        {
            std::cout << "__POWER_NODE__Forward:" << e.what() << std::endl;
        }
    }

    void Backward()
    {
        try
        {
            auto tem = this->pre_node[0]->data;

            this->pre_node[0]->grad += this->back_node->grad * (this->pre_node[1]->data * (pow(this->pre_node[0]->data, (this->pre_node[1]->data - 1))));

            this->pre_node[1]->grad += this->back_node->grad * pow(tem, this->pre_node[1]->data) * log(tem);
        }
        catch (std::exception &e)
        {
            std::cout << "__POWER_NODE__Backward:" << e.what() << std::endl;
        }
    }
};

template <typename T>
class __EXP_NODE__ : public __CAL_NODE__<T>
{
public:
    void AddPreNode(__DATA_NODE__<T> *pre_node)
    {
        try
        {
            if (this->pre_node.size() >= 1)
                throw "The \"pre_node\" of \"__EXP_NODE__\" has been overflowed!!";
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
                throw "The value of length of the \"__EXP_NODE__\" need to be 1,but the actual value of it is not enough!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"baak_node\" of \"__EXP_NODE__\" is nullptr!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            this->back_node->data = exp(this->pre_node[0]->data);

            this->if_forward.clear();
        }
        catch (std::exception &e)
        {
            std::cout << "__EXP_NODE__Forward:" << e.what() << std::endl;
        }
    }

    void Backward()
    {
        try
        {
            this->pre_node[0]->grad += this->back_node->grad * exp(this->pre_node[0]->data);
        }
        catch (std::exception &e)
        {
            std::cout << "__EXP_NODE__Backward:" << e.what() << std::endl;
        }
    }
};

template <typename T>
class __LN_NODE__ : public __CAL_NODE__<T>
{
public:
    void AddPreNode(__DATA_NODE__<T> *pre_node)
    {
        try
        {
            if (this->pre_node.size() >= 1)
                throw "The \"pre_node\" of \"__LN_NODE__\" has been overflowed!!";
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
                throw "The value of length of the \"__LN_NODE__\" need to be 1,but the actual value of it is not enough!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"baak_node\" of \"__LN_NODE__\" is nullptr!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            this->back_node->data = log(this->pre_node[0]->data);

            this->if_forward.clear();
        }
        catch (std::exception &e)
        {
            std::cout << "__LN_NODE__Forward:" << e.what() << std::endl;
        }
    }

    void Backward()
    {
        try
        {
            this->pre_node[0]->grad += this->back_node->grad * (1 / this->pre_node[0]->data);
        }
        catch (std::exception &e)
        {
            std::cout << "__LN_NODE__Backward:" << e.what() << std::endl;
        }
    }
};

template <typename T>
class __SIN_NODE__ : public __CAL_NODE__<T>
{
public:
    void AddPreNode(__DATA_NODE__<T> *pre_node)
    {
        try
        {
            if (this->pre_node.size() >= 1)
                throw "The \"pre_node\" of \"__SIN_NODE__\" has been overflowed!!";
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
                throw "The value of length of the \"__SIN_NODE__\" need to be 1,but the actual value of it is not enough!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"baak_node\" of \"__SIN_NODE__\" is nullptr!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            this->back_node->data = sin(this->pre_node[0]->data);

            this->if_forward.clear();
        }
        catch (std::exception &e)
        {
            std::cout << "__SIN_NODE__Forward:" << e.what() << std::endl;
        }
    }

    void Backward()
    {
        try
        {
            this->pre_node[0]->grad += this->back_node->grad * cos(this->pre_node[0]->data);
        }
        catch (std::exception &e)
        {
            std::cout << "__SIN_NODE__Backward:" << e.what() << std::endl;
        }
    }
};

template <typename T>
class __COS_NODE__ : public __CAL_NODE__<T>
{
public:
    void AddPreNode(__DATA_NODE__<T> *pre_node)
    {
        try
        {
            if (this->pre_node.size() >= 1)
                throw "The \"pre_node\" of \"__COS_NODE__\" has been overflowed!!";
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
                throw "The value of length of the \"__COS_NODE__\" need to be 1,but the actual value of it is not enough!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"baak_node\" of \"__COS_NODE__\" is nullptr!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            this->back_node->data = cos(this->pre_node[0]->data);

            this->if_forward.clear();
        }
        catch (std::exception &e)
        {
            std::cout << "__COS_NODE__Forward:" << e.what() << std::endl;
        }
    }

    void Backward()
    {
        try
        {
            this->pre_node[0]->grad += -this->back_node->grad * sin(this->pre_node[0]->data);
        }
        catch (std::exception &e)
        {
            std::cout << "__COS_NODE__Backward:" << e.what() << std::endl;
        }
    }
};

template <typename T>
class __TAN_NODE__ : public __CAL_NODE__<T>
{
public:
    void AddPreNode(__DATA_NODE__<T> *pre_node)
    {
        try
        {
            if (this->pre_node.size() >= 1)
                throw "The \"pre_node\" of \"__TAN_NODE__\" has been overflowed!!";
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
                throw "The value of length of the \"__TAN_NODE__\" need to be 1,but the actual value of it is not enough!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"baak_node\" of \"__TAN_NODE__\" is nullptr!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            this->back_node->data = tan(this->pre_node[0]->data);

            this->if_forward.clear();
        }
        catch (std::exception &e)
        {
            std::cout << "__TAN_NODE__Forward:" << e.what() << std::endl;
        }
    }

    void Backward()
    {
        try
        {
            this->pre_node[0]->grad += this->back_node->grad * (1 + (tan(this->pre_node[0]->data)) * (tan(this->pre_node[0]->data)));
        }
        catch (std::exception &e)
        {
            std::cout << "__TAN_NODE__Backward:" << e.what() << std::endl;
        }
    }
};

#endif


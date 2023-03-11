#ifndef NORM
#define NORM
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

#define EPS 0.00001

//batch1d归一化的宏定义
#define BATCHNORM1D_DIM \
    {                   \
        0, 2            \
    }
#define BATCHNORM1D_BROADCAST newaxis(), all(), newaxis()
#define BATCHNORM1D_ELENUM (INT32)((INT32)this->pre_node[0]->data.shape(0) * (INT32)this->pre_node[0]->data.shape(2))

//batch2d归一化的宏定义
#define BATCHNORM2D_DIM \
    {                   \
        0, 2,3           \
    }
#define BATCHNORM2D_BROADCAST newaxis(), all(), newaxis(),newaxis()
#define BATCHNORM2D_ELENUM (INT32)((INT32)this->pre_node[0]->data.shape(0) * (INT32)this->pre_node[0]->data.shape(2)*(INT32)this->pre_node[0]->data.shape(3))

//layer1d归一化的宏定义
#define LAYERNORM1D_DIM \
    {                   \
        1,2           \
    }
#define LAYERNORM1D_BROADCAST all(), newaxis(),newaxis()
#define LAYERNORM1D_ELENUM (INT32)((INT32)this->pre_node[0]->data.shape(1) * (INT32)this->pre_node[0]->data.shape(2))

//layer2d归一化的宏定义
#define LAYERNORM2D_DIM \
    {                   \
        1, 2,3           \
    }
#define LAYERNORM2D_BROADCAST  all(), newaxis(),newaxis(),newaxis()
#define LAYERNORM2D_ELENUM (INT32)((INT32)this->pre_node[0]->data.shape(1) * (INT32)this->pre_node[0]->data.shape(2)*(INT32)this->pre_node[0]->data.shape(3))


//instance2d归一化的宏定义
#define INSTANCENORM1D_DIM \
    {                   \
        1, 2,3           \
    }
#define INSTANCENORM1D_BROADCAST  all(), newaxis(),newaxis(),newaxis()
#define INSTANCENORM1D_ELENUM (INT32)((INT32)this->pre_node[0]->data.shape(1) * (INT32)this->pre_node[0]->data.shape(2)*(INT32)this->pre_node[0]->data.shape(3))

//instance2d归一化的宏定义
#define INSTANCENORM2D_DIM \
    {                   \
        1, 2,3           \
    }
#define INSTANCENORM2D_BROADCAST  all(), newaxis(),newaxis(),newaxis()
#define INSTANCENORM2D_ELENUM (INT32)((INT32)this->pre_node[0]->data.shape(1) * (INT32)this->pre_node[0]->data.shape(2)*(INT32)this->pre_node[0]->data.shape(3))




#define BATCHNORM1D_NODE_FLOAT32 __BATCHNORM1D_NODE__<FLOAT32>
#define BATCHNORM2D_NODE_FLOAT32 __BATCHNORM2D_NODE__<FLOAT32>
#define LAYERNORM1D_NODE_FLOAT32 __LAYERNORM1D_NODE__<FLOAT32>
#define LAYERNORM2D_NODE_FLOAT32 __LAYERNORM2D_NODE__<FLOAT32>
#define INSTANCENORM1D_NODE_FLOAT32 __INSTANCENORM1D_NODE__<FLOAT32>
#define INSTANCENORM2D_NODE_FLOAT32 __INSTANCENORM2D_NODE__<FLOAT32>

template <typename T>
class __BATCHNORM1D_NODE__ : public __CAL_NODE__<T>
{
public:
    /**
     *
     *   输入张量的形状是(N,C,L)
     *
     *
     */
    void Forward()
    {
        try
        {
            if (this->pre_node.size() < 1)
                throw "The value of length of the \"__BATCHNORM1D_NODE__\" need to be 2,but the actual value of it is not enough!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"back_node\" of \"__BATCHNORM1D_NODE__\" is nullptr!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            // 正向传播

            auto E_x = mean(this->pre_node[0]->data, BATCHNORM1D_DIM);
            //(C,)
            auto V_x = view(sum(pow(this->pre_node[0]->data - view(E_x, BATCHNORM1D_BROADCAST), 2), BATCHNORM1D_DIM), BATCHNORM1D_BROADCAST) / BATCHNORM1D_ELENUM;
            //(1,C,1)
            this->back_node->data = (this->pre_node[0]->data - view(E_x, BATCHNORM1D_BROADCAST)) / sqrt(V_x + EPS);

            // std::cout << this->back_node->data << std::endl;

            // std::cout << this->bias << std::endl;
            // std::cout << this->back_node->data << std::endl;

            // auto useless = 0;

            this->if_forward.clear();
        }
        catch (std::exception &e)
        {
            std::cout << "__BATCHNORM1D_NODE__Forward:" << e.what() << std::endl;
        }
    }

    void Backward()
    {
        try
        {
            // 反向传播
            auto input = this->pre_node[0]->data;
            auto E_x = mean(this->pre_node[0]->data, BATCHNORM1D_DIM);//(c,)
            auto V_x = view(sum(pow(this->pre_node[0]->data - view(E_x, BATCHNORM1D_BROADCAST), 2), BATCHNORM1D_DIM), BATCHNORM1D_BROADCAST) / BATCHNORM1D_ELENUM; //(1,c,1)

            

            auto A1 = this->back_node->grad * (xarray<T>({ 1 }) / sqrt(V_x + EPS));

            auto A2 = view(sum(this->back_node->grad * (xarray<T>({ 1 }) / BATCHNORM1D_ELENUM) / sqrt(V_x + EPS), BATCHNORM1D_DIM), BATCHNORM1D_BROADCAST);

            auto a1 = (xarray<T>({ 2 }) / BATCHNORM1D_ELENUM) * input;
            auto a2 = (xarray<T>({ 2 }) / BATCHNORM1D_ELENUM) * view(E_x, BATCHNORM1D_BROADCAST);
            auto a3 = view((xarray<T>({ 2 }) / (BATCHNORM1D_ELENUM * BATCHNORM1D_ELENUM)) * sum(this->pre_node[0]->data - view(E_x, BATCHNORM1D_BROADCAST), BATCHNORM1D_DIM), BATCHNORM1D_BROADCAST);
            auto Vx_dev = a1 - a2 - a3;


            auto b1 = xarray<T>({ 0.5 }) * pow((V_x + EPS), -0.5) * (xarray<T>({ 1 }) / (V_x + EPS));
            auto b2 = input - view(E_x, BATCHNORM1D_BROADCAST);
            auto A3 = Vx_dev * view(sum((this->back_node->grad * b1 * b2), BATCHNORM1D_DIM), BATCHNORM1D_BROADCAST);


            this->pre_node[0]->grad = A1 - A2 - A3;



          /*  std::cout << A1<<std::endl;
            std::cout << A2 << std::endl;
            std::cout << A3 << std::endl;*/




            auto useless = 0;
        }
        catch (std::exception &e)
        {
            std::cout << "__BATCHNORM1D_NODE__Backward:" << e.what() << std::endl;
        }
    }
};

template <typename T>
class __BATCHNORM2D_NODE__ : public __CAL_NODE__<T>
{
public:
    /**
     *
     *   输入张量的形状是(N,C,W,H)
     *
     *
     */
    void Forward()
    {
        try
        {
            if (this->pre_node.size() < 1)
                throw "The value of length of the \"__BATCHNORM2D_NODE__\" need to be 2,but the actual value of it is not enough!";
        }
        catch (const char* s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"back_node\" of \"__BATCHNORM2D_NODE__\" is nullptr!";
        }
        catch (const char* s)
        {
            std::cout << s << '\n';
        }

        try
        {
            // 正向传播

            auto E_x = mean(this->pre_node[0]->data, BATCHNORM2D_DIM);
            //(C,)
            auto V_x = view(sum(pow(this->pre_node[0]->data - view(E_x, BATCHNORM2D_BROADCAST), 2), BATCHNORM2D_DIM), BATCHNORM2D_BROADCAST) / BATCHNORM2D_ELENUM;
            //(1,C,1)
            this->back_node->data = (this->pre_node[0]->data - view(E_x, BATCHNORM2D_BROADCAST)) / sqrt(V_x + EPS);

            // std::cout << this->back_node->data << std::endl;

            // std::cout << this->bias << std::endl;
            // std::cout << this->back_node->data << std::endl;

            // auto useless = 0;

            this->if_forward.clear();
        }
        catch (std::exception& e)
        {
            std::cout << "__BATCHNORM2D_NODE__Forward:" << e.what() << std::endl;
        }
    }

    void Backward()
    {
        try
        {
            // 反向传播
            auto input = this->pre_node[0]->data;
            auto E_x = mean(this->pre_node[0]->data, BATCHNORM2D_DIM);//(c,)
            auto V_x = view(sum(pow(this->pre_node[0]->data - view(E_x, BATCHNORM2D_BROADCAST), 2), BATCHNORM2D_DIM), BATCHNORM2D_BROADCAST) / BATCHNORM2D_ELENUM; //(1,c,1)



            auto A1 = this->back_node->grad * (xarray<T>({ 1 }) / sqrt(V_x + EPS));

            auto A2 = view(sum(this->back_node->grad * (xarray<T>({ 1 }) / BATCHNORM2D_ELENUM) / sqrt(V_x + EPS), BATCHNORM2D_DIM), BATCHNORM2D_BROADCAST);

            auto a1 = (xarray<T>({ 2 }) / BATCHNORM2D_ELENUM) * input;
            auto a2 = (xarray<T>({ 2 }) / BATCHNORM2D_ELENUM) * view(E_x, BATCHNORM2D_BROADCAST);
            auto a3 = view((xarray<T>({ 2 }) / (BATCHNORM2D_ELENUM * BATCHNORM2D_ELENUM)) * sum(this->pre_node[0]->data - view(E_x, BATCHNORM2D_BROADCAST), BATCHNORM2D_DIM), BATCHNORM2D_BROADCAST);
            auto Vx_dev = a1 - a2 - a3;


            auto b1 = xarray<T>({ 0.5 }) * pow((V_x + EPS), -0.5) * (xarray<T>({ 1 }) / (V_x + EPS));
            auto b2 = input - view(E_x, BATCHNORM2D_BROADCAST);
            auto A3 = Vx_dev * view(sum((this->back_node->grad * b1 * b2), BATCHNORM2D_DIM), BATCHNORM2D_BROADCAST);


            this->pre_node[0]->grad = A1 - A2 - A3;



            /*std::cout << this->pre_node[0]->grad << std::endl;*/
            /*std::cout << A2 << std::endl;
            std::cout << A3 << std::endl;*/




            auto useless = 0;
        }
        catch (std::exception& e)
        {
            std::cout << "__BATCHNORM2D_NODE__Backward:" << e.what() << std::endl;
        }
    }
};


template <typename T>
class __LAYERNORM1D_NODE__ : public __CAL_NODE__<T>
{
public:
    /**
     *
     *   输入张量的形状是(N,C,W,H)
     *
     *
     */
    void Forward()
    {
        try
        {
            if (this->pre_node.size() < 1)
                throw "The value of length of the \"__LAYERNORM1D_NODE__\" need to be 2,but the actual value of it is not enough!";
        }
        catch (const char* s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"back_node\" of \"__LAYERNORM1D_NODE__\" is nullptr!";
        }
        catch (const char* s)
        {
            std::cout << s << '\n';
        }

        try
        {
            // 正向传播

            auto E_x = mean(this->pre_node[0]->data, LAYERNORM1D_DIM);
            //(C,)
            auto V_x = view(sum(pow(this->pre_node[0]->data - view(E_x, LAYERNORM1D_BROADCAST), 2), LAYERNORM1D_DIM), LAYERNORM1D_BROADCAST) / LAYERNORM1D_ELENUM;
            //(1,C,1)
            this->back_node->data = (this->pre_node[0]->data - view(E_x, LAYERNORM1D_BROADCAST)) / sqrt(V_x + EPS);

            // std::cout << this->back_node->data << std::endl;

            // std::cout << this->bias << std::endl;
            // std::cout << this->back_node->data << std::endl;

            // auto useless = 0;

            this->if_forward.clear();
        }
        catch (std::exception& e)
        {
            std::cout << "__LAYERNORM1D_NODE__Forward:" << e.what() << std::endl;
        }
    }

    void Backward()
    {
        try
        {
            // 反向传播
            auto input = this->pre_node[0]->data;
            auto E_x = mean(this->pre_node[0]->data, LAYERNORM1D_DIM);//(c,)
            auto V_x = view(sum(pow(this->pre_node[0]->data - view(E_x, LAYERNORM1D_BROADCAST), 2), LAYERNORM1D_DIM), LAYERNORM1D_BROADCAST) / LAYERNORM1D_ELENUM; //(1,c,1)



            auto A1 = this->back_node->grad * (xarray<T>({ 1 }) / sqrt(V_x + EPS));

            auto A2 = view(sum(this->back_node->grad * (xarray<T>({ 1 }) / LAYERNORM1D_ELENUM) / sqrt(V_x + EPS), LAYERNORM1D_DIM), LAYERNORM1D_BROADCAST);

            auto a1 = (xarray<T>({ 2 }) / LAYERNORM1D_ELENUM) * input;
            auto a2 = (xarray<T>({ 2 }) / LAYERNORM1D_ELENUM) * view(E_x, LAYERNORM1D_BROADCAST);
            auto a3 = view((xarray<T>({ 2 }) / (LAYERNORM1D_ELENUM * LAYERNORM1D_ELENUM)) * sum(this->pre_node[0]->data - view(E_x, LAYERNORM1D_BROADCAST), LAYERNORM1D_DIM), LAYERNORM1D_BROADCAST);
            auto Vx_dev = a1 - a2 - a3;


            auto b1 = xarray<T>({ 0.5 }) * pow((V_x + EPS), -0.5) * (xarray<T>({ 1 }) / (V_x + EPS));
            auto b2 = input - view(E_x, LAYERNORM1D_BROADCAST);
            auto A3 = Vx_dev * view(sum((this->back_node->grad * b1 * b2), LAYERNORM1D_DIM), LAYERNORM1D_BROADCAST);


            this->pre_node[0]->grad = A1 - A2 - A3;



            /*std::cout << this->pre_node[0]->grad << std::endl;*/
            /*std::cout << A2 << std::endl;
            std::cout << A3 << std::endl;*/




            auto useless = 0;
        }
        catch (std::exception& e)
        {
            std::cout << "__LAYERNORM1D_NODE__Backward:" << e.what() << std::endl;
        }
    }
};



template <typename T>
class __LAYERNORM2D_NODE__ : public __CAL_NODE__<T>
{
public:
    /**
     *
     *   输入张量的形状是(N,C,W,H)
     *
     *
     */
    void Forward()
    {
        try
        {
            if (this->pre_node.size() < 1)
                throw "The value of length of the \"__LAYERNORM2D_NODE__\" need to be 2,but the actual value of it is not enough!";
        }
        catch (const char* s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"back_node\" of \"__LAYERNORM2D_NODE__\" is nullptr!";
        }
        catch (const char* s)
        {
            std::cout << s << '\n';
        }

        try
        {
            // 正向传播

            auto E_x = mean(this->pre_node[0]->data, LAYERNORM2D_DIM);
            //(C,)
            auto V_x = view(sum(pow(this->pre_node[0]->data - view(E_x, LAYERNORM2D_BROADCAST), 2), LAYERNORM2D_DIM), LAYERNORM2D_BROADCAST) / LAYERNORM2D_ELENUM;
            //(1,C,1)
            this->back_node->data = (this->pre_node[0]->data - view(E_x, LAYERNORM2D_BROADCAST)) / sqrt(V_x + EPS);

            // std::cout << this->back_node->data << std::endl;

            // std::cout << this->bias << std::endl;
            // std::cout << this->back_node->data << std::endl;

            // auto useless = 0;

            this->if_forward.clear();
        }
        catch (std::exception& e)
        {
            std::cout << "__LAYERNORM2D_NODE__Forward:" << e.what() << std::endl;
        }
    }

    void Backward()
    {
        try
        {
            // 反向传播
            auto input = this->pre_node[0]->data;
            auto E_x = mean(this->pre_node[0]->data, LAYERNORM2D_DIM);//(c,)
            auto V_x = view(sum(pow(this->pre_node[0]->data - view(E_x, LAYERNORM2D_BROADCAST), 2), LAYERNORM2D_DIM), LAYERNORM2D_BROADCAST) / LAYERNORM2D_ELENUM; //(1,c,1)



            auto A1 = this->back_node->grad * (xarray<T>({ 1 }) / sqrt(V_x + EPS));

            auto A2 = view(sum(this->back_node->grad * (xarray<T>({ 1 }) / LAYERNORM2D_ELENUM) / sqrt(V_x + EPS), LAYERNORM2D_DIM), LAYERNORM2D_BROADCAST);

            auto a1 = (xarray<T>({ 2 }) / LAYERNORM2D_ELENUM) * input;
            auto a2 = (xarray<T>({ 2 }) / LAYERNORM2D_ELENUM) * view(E_x, LAYERNORM2D_BROADCAST);
            auto a3 = view((xarray<T>({ 2 }) / (LAYERNORM2D_ELENUM * LAYERNORM2D_ELENUM)) * sum(this->pre_node[0]->data - view(E_x, LAYERNORM2D_BROADCAST), LAYERNORM2D_DIM), LAYERNORM2D_BROADCAST);
            auto Vx_dev = a1 - a2 - a3;


            auto b1 = xarray<T>({ 0.5 }) * pow((V_x + EPS), -0.5) * (xarray<T>({ 1 }) / (V_x + EPS));
            auto b2 = input - view(E_x, LAYERNORM2D_BROADCAST);
            auto A3 = Vx_dev * view(sum((this->back_node->grad * b1 * b2), LAYERNORM2D_DIM), LAYERNORM2D_BROADCAST);


            this->pre_node[0]->grad = A1 - A2 - A3;



            std::cout << this->pre_node[0]->grad << std::endl;
            /*std::cout << A2 << std::endl;
            std::cout << A3 << std::endl;*/




            auto useless = 0;
        }
        catch (std::exception& e)
        {
            std::cout << "__LAYERNORM2D_NODE__Backward:" << e.what() << std::endl;
        }
    }
};



template <typename T>
class __INSTANCENORM1D_NODE__ : public __CAL_NODE__<T>
{
public:
    /**
     *
     *   输入张量的形状是(N,C,W,H)
     *
     *
     */
    void Forward()
    {
        try
        {
            if (this->pre_node.size() < 1)
                throw "The value of length of the \"__INSTANCENORM1D_NODE__\" need to be 2,but the actual value of it is not enough!";
        }
        catch (const char* s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"back_node\" of \"__INSTANCENORM1D_NODE__\" is nullptr!";
        }
        catch (const char* s)
        {
            std::cout << s << '\n';
        }

        try
        {
            // 正向传播

            auto E_x = mean(this->pre_node[0]->data, INSTANCENORM1D_DIM);
            //(C,)
            auto V_x = view(sum(pow(this->pre_node[0]->data - view(E_x, INSTANCENORM1D_BROADCAST), 2), INSTANCENORM1D_DIM), INSTANCENORM1D_BROADCAST) / INSTANCENORM1D_ELENUM;
            //(1,C,1)
            this->back_node->data = (this->pre_node[0]->data - view(E_x, INSTANCENORM1D_BROADCAST)) / sqrt(V_x + EPS);

            // std::cout << this->back_node->data << std::endl;

            // std::cout << this->bias << std::endl;
            // std::cout << this->back_node->data << std::endl;

            // auto useless = 0;

            this->if_forward.clear();
        }
        catch (std::exception& e)
        {
            std::cout << "__INSTANCENORM1D_NODE__Forward:" << e.what() << std::endl;
        }
    }

    void Backward()
    {
        try
        {
            // 反向传播
            auto input = this->pre_node[0]->data;
            auto E_x = mean(this->pre_node[0]->data, INSTANCENORM1D_DIM);//(c,)
            auto V_x = view(sum(pow(this->pre_node[0]->data - view(E_x, INSTANCENORM1D_BROADCAST), 2), INSTANCENORM1D_DIM), INSTANCENORM1D_BROADCAST) / INSTANCENORM1D_ELENUM; //(1,c,1)



            auto A1 = this->back_node->grad * (xarray<T>({ 1 }) / sqrt(V_x + EPS));

            auto A2 = view(sum(this->back_node->grad * (xarray<T>({ 1 }) / INSTANCENORM1D_ELENUM) / sqrt(V_x + EPS), INSTANCENORM1D_DIM), INSTANCENORM1D_BROADCAST);

            auto a1 = (xarray<T>({ 2 }) / INSTANCENORM1D_ELENUM) * input;
            auto a2 = (xarray<T>({ 2 }) / INSTANCENORM1D_ELENUM) * view(E_x, INSTANCENORM1D_BROADCAST);
            auto a3 = view((xarray<T>({ 2 }) / (INSTANCENORM1D_ELENUM * INSTANCENORM1D_ELENUM)) * sum(this->pre_node[0]->data - view(E_x, INSTANCENORM1D_BROADCAST), INSTANCENORM1D_DIM), INSTANCENORM1D_BROADCAST);
            auto Vx_dev = a1 - a2 - a3;


            auto b1 = xarray<T>({ 0.5 }) * pow((V_x + EPS), -0.5) * (xarray<T>({ 1 }) / (V_x + EPS));
            auto b2 = input - view(E_x, INSTANCENORM1D_BROADCAST);
            auto A3 = Vx_dev * view(sum((this->back_node->grad * b1 * b2), INSTANCENORM1D_DIM), INSTANCENORM1D_BROADCAST);


            this->pre_node[0]->grad = A1 - A2 - A3;



            /*std::cout << this->pre_node[0]->grad << std::endl;*/
            /*std::cout << A2 << std::endl;
            std::cout << A3 << std::endl;*/




            auto useless = 0;
        }
        catch (std::exception& e)
        {
            std::cout << "__INSTANCENORM1D_NODE__Backward:" << e.what() << std::endl;
        }
    }
};





template <typename T>
class __INSTANCENORM2D_NODE__ : public __CAL_NODE__<T>
{
public:
    /**
     *
     *   输入张量的形状是(N,C,W,H)
     *
     *
     */
    void Forward()
    {
        try
        {
            if (this->pre_node.size() < 1)
                throw "The value of length of the \"__INSTANCENORM2D_NODE__\" need to be 2,but the actual value of it is not enough!";
        }
        catch (const char* s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"back_node\" of \"__INSTANCENORM2D_NODE__\" is nullptr!";
        }
        catch (const char* s)
        {
            std::cout << s << '\n';
        }

        try
        {
            // 正向传播

            auto E_x = mean(this->pre_node[0]->data, INSTANCENORM2D_DIM);
            //(C,)
            auto V_x = view(sum(pow(this->pre_node[0]->data - view(E_x, INSTANCENORM2D_BROADCAST), 2), INSTANCENORM2D_DIM), INSTANCENORM2D_BROADCAST) / INSTANCENORM2D_ELENUM;
            //(1,C,1)
            this->back_node->data = (this->pre_node[0]->data - view(E_x, INSTANCENORM2D_BROADCAST)) / sqrt(V_x + EPS);

            // std::cout << this->back_node->data << std::endl;

            // std::cout << this->bias << std::endl;
            // std::cout << this->back_node->data << std::endl;

            // auto useless = 0;

            this->if_forward.clear();
        }
        catch (std::exception& e)
        {
            std::cout << "__INSTANCENORM2D_NODE__Forward:" << e.what() << std::endl;
        }
    }

    void Backward()
    {
        try
        {
            // 反向传播
            auto input = this->pre_node[0]->data;
            auto E_x = mean(this->pre_node[0]->data, INSTANCENORM2D_DIM);//(c,)
            auto V_x = view(sum(pow(this->pre_node[0]->data - view(E_x, INSTANCENORM2D_BROADCAST), 2), INSTANCENORM2D_DIM), INSTANCENORM2D_BROADCAST) / INSTANCENORM2D_ELENUM; //(1,c,1)



            auto A1 = this->back_node->grad * (xarray<T>({ 1 }) / sqrt(V_x + EPS));

            auto A2 = view(sum(this->back_node->grad * (xarray<T>({ 1 }) / INSTANCENORM2D_ELENUM) / sqrt(V_x + EPS), INSTANCENORM2D_DIM), INSTANCENORM2D_BROADCAST);

            auto a1 = (xarray<T>({ 2 }) / INSTANCENORM2D_ELENUM) * input;
            auto a2 = (xarray<T>({ 2 }) / INSTANCENORM2D_ELENUM) * view(E_x, INSTANCENORM2D_BROADCAST);
            auto a3 = view((xarray<T>({ 2 }) / (INSTANCENORM2D_ELENUM * INSTANCENORM2D_ELENUM)) * sum(this->pre_node[0]->data - view(E_x, INSTANCENORM2D_BROADCAST), INSTANCENORM2D_DIM), INSTANCENORM2D_BROADCAST);
            auto Vx_dev = a1 - a2 - a3;


            auto b1 = xarray<T>({ 0.5 }) * pow((V_x + EPS), -0.5) * (xarray<T>({ 1 }) / (V_x + EPS));
            auto b2 = input - view(E_x, INSTANCENORM2D_BROADCAST);
            auto A3 = Vx_dev * view(sum((this->back_node->grad * b1 * b2), INSTANCENORM2D_DIM), INSTANCENORM2D_BROADCAST);


            this->pre_node[0]->grad = A1 - A2 - A3;



          /*  std::cout << this->pre_node[0]->grad << std::endl;*/
            /*std::cout << A2 << std::endl;
            std::cout << A3 << std::endl;*/




            auto useless = 0;
        }
        catch (std::exception& e)
        {
            std::cout << "__INSTANCENORM2D_NODE__Backward:" << e.what() << std::endl;
        }
    }
};




#endif
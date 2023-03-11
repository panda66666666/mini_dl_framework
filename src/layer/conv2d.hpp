#ifndef CONV2D
#define CONV2D
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
#include <xtensor/xpad.hpp>
#include <xtensor/xbroadcast.hpp>
#include <exception>
#include <vector>
#include <queue>
#include <string>
#include <math.h>

using namespace xt;

#define CONV2D_NODE_FLOAT32 __CONV2D_NODE__<FLOAT32>

template <typename T>
class __CONV2D_NODE__ : public __CAL_NODE__<T>
{
public:
    UNSIGNED_INT32 in_channels;
    UNSIGNED_INT32 out_channel;
    UNSIGNED_INT32 kernel_size;
    UNSIGNED_INT32 stride;
    UNSIGNED_INT32 padding;
    bool need_bias;

    xarray<T> kernel;
    xarray<T> kernel_grad;

    xarray<T> bias;
    xarray<T> bias_grad;

    __CONV2D_NODE__(UNSIGNED_INT32 in_channels, UNSIGNED_INT32 out_channel, UNSIGNED_INT32 kernel_size, UNSIGNED_INT32 stride = 1, UNSIGNED_INT32 padding = 0, bool need_bias = true)
    {
        xt::random::seed(time(NULL));

        this->in_channels = in_channels;
        this->out_channel = out_channel;
        this->kernel_size = kernel_size;
        this->stride = stride;
        this->padding = padding;
        this->need_bias = need_bias;

        FLOAT32 LIM = 1 / sqrt((FLOAT32)in_channels * kernel_size * kernel_size);
        // this->para=

         this->kernel = random::rand<T>({this->in_channels, this->out_channel,
                                         this->kernel_size, this->kernel_size},
                                        -LIM, LIM);

        /*this->kernel = ones<T>({this->in_channels, this->out_channel,
                                this->kernel_size, this->kernel_size});*/
        this->kernel_grad = zeros<T>({this->in_channels, this->out_channel,
                                      this->kernel_size, this->kernel_size});

        if (need_bias)
        {
             this->bias = random::rand<T>({this->out_channel}, -LIM, LIM);

           /* this->bias = ones<T>({this->out_channel});*/
            this->bias_grad = zeros<T>({this->out_channel});
        }
        else
        {
            this->bias = DEFAULT_TENSOR_INT16;
            this->bias_grad = DEFAULT_TENSOR_INT16;
        }

        auto useless = 0;
    }

    void AddPreNode(__DATA_NODE__<T> *pre_node)
    {
        try
        {
            if (this->pre_node.size() >= 1)
                throw "The \"pre_node\" of \"__CONV2D_NODE__\" has been overflowed!!";
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
                throw "The value of length of the \"__CONV2D_NODE__\" need to be 2,but the actual value of it is not enough!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"back_node\" of \"__CONV2D_NODE__\" is nullptr!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if ((UNSIGNED_INT32)this->pre_node[0]->data.shape(1) != this->in_channels)
                throw "The input`s channel is not matching to the paramaters!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            // 正向传播
            // auto input = this->pre_node[0]->data;

            UNSIGNED_INT32 input_w = this->pre_node[0]->data.shape(2);
            UNSIGNED_INT32 input_h = this->pre_node[0]->data.shape(3);

            UNSIGNED_INT32 out_w = (UNSIGNED_INT32)((input_w + 2 * this->padding - this->kernel_size) / this->stride + 1);
            UNSIGNED_INT32 out_h = (UNSIGNED_INT32)((input_h + 2 * this->padding - this->kernel_size) / this->stride + 1);

            xarray<T> input_pad = pad(this->pre_node[0]->data, {{0, 0}, {0, 0}, {this->padding, this->padding}, {this->padding, this->padding}}, pad_mode::constant, 0);

            xarray<T> output = zeros<T>({(UNSIGNED_INT32)this->pre_node[0]->data.shape(0), this->out_channel, out_w, out_h});

            // std::cout << input_pad_tensor << std::endl;
            // std::cout << this->kernel << std::endl;

            for (UNSIGNED_INT32 w = 0; w < out_w; w++)
                for (UNSIGNED_INT32 h = 0; h < out_h; h++)
                {
                    // 定义当前卷积核对应的开始和结束的位置
                    UNSIGNED_INT32 w_start = w * this->stride;
                    UNSIGNED_INT32 w_end = w_start + this->kernel_size;
                    UNSIGNED_INT32 h_start = h * this->stride;
                    UNSIGNED_INT32 h_end = h_start + this->kernel_size;

                    // std::cout << view(output_tensor,all(),all(),w,h) << std::endl;

                    // std::cout <<view(this->kernel, all(), all(), all(), all()) *
                    // view(input_pad_tensor, all(), all(), range(w_start, w_end), range(h_start, h_end))
                    // std::cout << adapt(view(input_pad, all(), all(), newaxis(), range(w_start, w_end), range(h_start, h_end)).shape());

                    view(output, all(), all(), w, h) =

                        sum(view(input_pad, all(), all(), newaxis(), range(w_start, w_end), range(h_start, h_end)) *

                                broadcast(view(this->kernel, newaxis(), all()), {(UNSIGNED_INT32)this->pre_node[0]->data.shape(0), this->in_channels, this->out_channel, this->kernel_size, this->kernel_size}),

                            {1, 3, 4});

                    auto useless = 0;
                }

            if (this->need_bias)
            {
                output += view(this->bias, newaxis(), all(), newaxis(), newaxis());
            }

            this->back_node->data = output;
            // std::cout << this->back_node->data;

            this->if_forward.clear();
        }
        catch (std::exception &e)
        {

            std::cout << "__CONV2D_NODE__Forward:" << e.what() << std::endl;
        }
    }

    void Backward()
    {
        try
        {
            // 反向传播
            // auto output_grad = this->back_node->grad;

            UNSIGNED_INT32 input_w = this->pre_node[0]->data.shape(2);
            UNSIGNED_INT32 input_h = this->pre_node[0]->data.shape(3);

            UNSIGNED_INT32 out_w = (UNSIGNED_INT32)((input_w + 2 * this->padding - this->kernel_size) / this->stride + 1);
            UNSIGNED_INT32 out_h = (UNSIGNED_INT32)((input_h + 2 * this->padding - this->kernel_size) / this->stride + 1);

            xarray<T> input_grad = pad(zeros<T>(this->pre_node[0]->data.shape()), {{0, 0}, {0, 0}, {this->padding, this->padding}, {this->padding, this->padding}}, pad_mode::constant, 0);

            xarray<T> input_pad = pad(this->pre_node[0]->data, {{0, 0}, {0, 0}, {this->padding, this->padding}, {this->padding, this->padding}}, pad_mode::constant, 0);

            for (UNSIGNED_INT32 w = 0; w < out_w; w++)
                for (UNSIGNED_INT32 h = 0; h < out_h; h++)
                {
                    // 定义当前卷积核对应的开始和结束的位置
                    UNSIGNED_INT32 w_start = w * this->stride;
                    UNSIGNED_INT32 w_end = w_start + this->kernel_size;
                    UNSIGNED_INT32 h_start = h * this->stride;
                    UNSIGNED_INT32 h_end = h_start + this->kernel_size;

                    // std::cout << h<<adapt(sum(view(this->back_node->grad, all(), newaxis(), all(), range(w, w + 1), range(h, h + 1)) *

                    //             view(input_pad, all(), all(), newaxis(), range(w_start, w_end), range(h_start, h_end)),

                    //         0).shape());
                    // std::cout << adapt(this->kernel_grad.shape());

                    this->kernel_grad +=

                        sum(view(this->back_node->grad, all(), newaxis(), all(), range(w, w + 1), range(h, h + 1)) *

                                view(input_pad, all(), all(), newaxis(), range(w_start, w_end), range(h_start, h_end)),

                            0);

                    view(input_grad, all(), all(), range(w_start, w_end), range(h_start, h_end)) +=

                        sum(view(this->kernel, newaxis(), all()) *

                                view(this->back_node->grad, all(), newaxis(), all(), range(w, w + 1), range(h, h + 1)),

                            2);

                }
                

            this->pre_node[0]->grad += view(input_grad, all(), all(),
                                            range(this->padding, input_w + this->padding),
                                            range(this->padding, input_h + this->padding));

            if (this->need_bias)
                this->bias_grad += sum(this->back_node->grad, {0, 2, 3});

            // std::cout << this->pre_node[0]->grad;
            auto useless = 0;
        }
        catch (std::exception &e)
        {
            std::cout << "__CONV2D_NODE__Backward:" << e.what() << std::endl;
        }
    }
    void ZeroGrad() override
    {
        // std::cout << "执行到这里LE";
        this->kernel_grad = zeros<T>({this->in_channels, this->out_channel,
                                      this->kernel_size, this->kernel_size});

        if (this->need_bias)
            this->bias_grad = zeros<T>({this->out_channel});
    }

    void UpdatePara(T eta) override
    {
        // std::cout << "执行到这里!!";
        this->kernel -= eta * this->kernel_grad;
        if (this->need_bias)
            this->bias -= eta * this->bias_grad;
    }
};

#endif
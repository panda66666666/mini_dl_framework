#ifndef LINEAR
#define LINEAR
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

#define LINEAR_NODE_FLOAT32 __LINEAR_NODE__<FLOAT32>

template <typename T>
class __LINEAR_NODE__ : public __CAL_NODE__<T>
{
public:
    xarray<T> weights;
    xarray<T> weights_grad;
    xarray<T> bias;
    xarray<T> bias_grad;
    UNSIGNED_INT32 in_features;
    UNSIGNED_INT32 out_features;
    bool need_bias;

    __LINEAR_NODE__(UNSIGNED_INT32 in_features, UNSIGNED_INT32 out_features, bool need_bias = true)
    {
        xt::random::seed(time(NULL));

        this->need_bias = need_bias;
        this->in_features = in_features;
        this->out_features = out_features;
        FLOAT32 LIM = 1 / sqrt((FLOAT32)in_features);
        // this->para=

        this->weights = random::rand<T>({in_features, out_features}, -LIM, LIM);
        this->weights_grad = zeros<T>({in_features, out_features});

        if (need_bias)
        {
            this->bias = random::rand<T>({out_features}, -LIM, LIM);
            this->bias_grad = zeros<T>({out_features});
        }
        else
        {
            this->bias = DEFAULT_TENSOR_INT16;
            this->bias_grad = DEFAULT_TENSOR_INT16;
        }

        // std::cout << this->weights;
        auto useless = 0;
    }

    void AddPreNode(__DATA_NODE__<T> *pre_node)
    {
        try
        {
            if (this->pre_node.size() >= 1)
                throw "The \"pre_node\" of \"__LINEAR_NODE__\" has been overflowed!!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        this->pre_node.push_back(pre_node);
    }

    xarray<T> MultiDimMatMul(xarray<T> &input, xarray<T> &weights)
    {
        xarray<UNSIGNED_INT32>::shape_type shape = input.shape();
        *(shape.end() - 1) = this->out_features;

        xarray<T> out(shape);

        if (shape.size() == 2)
        {
            // std::cout << input << weights<<shape.size() << std::endl;
            out = matmul(input, weights);
            // std::cout << out << std::endl;
        }
        else
        {
            xt::xarray<T> input_slice;
            // int layer = shape.size();
            for (int i = 0; i < shape[1]; i++)
            {
                input_slice = view(input, all(), i, all());
                xt::view(out, all(), i, all()) = MultiDimMatMul(input_slice, weights);
            }
        }

        return out;
    }

    std::vector<xarray<T>> MultiDimBackward(const xarray<T> &input, const xarray<T> &input_grad, const xarray<T> &output, const xarray<T> &output_grad)
    {
        xarray<UNSIGNED_INT32>::shape_type shape = input.shape();

        std::vector<xarray<T>> out = {xarray<T>(input.shape()),
                                      xarray<T>(input_grad.shape()),
                                      xarray<T>(output.shape()),
                                      xarray<T>(output_grad.shape())};

        if (shape.size() == 1)
        {
            // std::cout << this->weights_grad << std::endl;
            // std::cout << this->bias_grad << std::endl;
            // std::cout << out[1] << std::endl;
            // 计算weights梯度
            this->weights_grad += ones<T>(this->weights.shape()) * view(input, all(), newaxis()) * output_grad;

            // 计算bias梯度
            this->bias_grad += output_grad;

            // 输入梯度计算
            /*std::cout << this->weights << std::endl;*/
          /*  std::cout << sum(this->weights * output_grad, 1);*/

            out[1] = sum(this->weights * output_grad, 1);

            // std::cout << this->weights_grad << std::endl;
            // std::cout << this->bias_grad << std::endl;
            // std::cout << out[1] << std::endl;

            // auto useless = 1;
        }
        else
        {

            for (UNSIGNED_INT32 i = 0; i < shape[0]; i++)
            {
                // std::cout << input << std::endl;
                // std::cout << input_grad << std::endl;
                // std::cout << output << std::endl;
                // std::cout << output_grad << std::endl;

                std::vector<xarray<T>> four_data =
                    MultiDimBackward(view(input, i, all()),
                                     view(input_grad, i, all()),
                                     view(output, i, all()),
                                     view(output_grad, i, all()));

                view(out[0], i, all()) = four_data[0];

      /*          std::cout << out[1] << std::endl;
                std::cout << four_data[1];*/

                view(out[1], i, all()) = four_data[1];
                view(out[2], i, all()) = four_data[2];
                view(out[3], i, all()) = four_data[3];
            }
        }

        return out;
    }

    void Forward()
    {
        try
        {
            if (this->pre_node.size() < 1)
                throw "The value of length of the \"__LINEAR_NODE__\" need to be 2,but the actual value of it is not enough!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"back_node\" of \"__LINEAR_NODE__\" is nullptr!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            // 正向传播
            if (this->need_bias)
                this->back_node->data = this->MultiDimMatMul(this->pre_node[0]->data, this->weights) + this->bias;
            else
                this->back_node->data = this->MultiDimMatMul(this->pre_node[0]->data, this->weights);
            // std::cout << this->weights << std::endl;
            // std::cout << this->bias << std::endl;
            // std::cout << this->back_node->data << std::endl;

            // auto useless = 0;

            this->if_forward.clear();
        }
        catch (std::exception &e)
        {
            std::cout << "__LINEAR_NODE__Forward:" << e.what() << std::endl;
        }
    }

    void Backward()
    {
        try
        {
            auto s = this->pre_node[0]->grad.shape();
            if (s.size() == 0 || s.size() == 1)
                this->pre_node[0]->grad = zeros<T>(this->pre_node[0]->data.shape());

            // std::cout << this->pre_node[0]->grad;

            std::vector<xarray<T>> four_data =
                MultiDimBackward(this->pre_node[0]->data,
                                 this->pre_node[0]->grad,
                                 this->back_node->data,
                                 this->back_node->grad);

            this->pre_node[0]->grad += four_data[1];
            // std::cout << four_data[1];
            // std::cout << this->pre_node[0]->grad << std::endl;
            // std::cout << this->weights_grad << std::endl;
            // std::cout << this->bias_grad << std::endl;

            auto useless = 0;
        }
        catch (std::exception &e)
        {
            std::cout << "__LINEAR_NODE__Backward:" << e.what() << std::endl;
        }
    }
    void ZeroGrad() override
    {
        // std::cout << "执行到这里LE";
        this->weights_grad = zeros<T>({this->in_features, this->out_features});
        if (this->need_bias)
            this->bias_grad = zeros<T>({this->out_features});
    }

    void UpdatePara(T eta) override
    {
        // std::cout << "执行到这里!!";
        this->weights -= eta * this->weights_grad;
        this->bias -= eta * this->bias_grad;
    }
};

#endif
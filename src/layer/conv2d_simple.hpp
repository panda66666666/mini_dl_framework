#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include <xtensor/xio.hpp>
#include<iostream>
#include<xtensor/xpad.hpp>
#include <xtensor/xfixed.hpp>
//#define FORCE_IMPORT_ARRAY
//#include "xtensor-python/pyarray.hpp"
using namespace xt;

class Conv_2d
{
    public:
        int in_channel;
        int out_channel;
        int kernel_size;
        int stride;
        int padding;
        xarray<float> input;
        xarray<float> kernel;

        Conv_2d(int in_channels, int out_channel, int kernel_size, int stride = 1,int padding=0)
        {
            this->in_channel = in_channels;
            this->out_channel = out_channel;
            this->kernel_size = kernel_size;
            this->stride = stride;
            this->padding = padding;
            this->kernel = random::randn<float>({ this->in_channel,this->out_channel,
                                                  this->kernel_size,this->kernel_size});
        }

        xarray<float> forward(xarray<float>& input_tensor)
        {
            int input_channel = this->in_channel; //输出通道大小
            int out_channel = this->out_channel; //输出通达大小
            
            int input_w = input_tensor.shape(2);
            int input_h = input_tensor.shape(3);

            int out_w = int((input_w + 2 * this->padding - this->kernel_size) / this->stride + 1);
            int out_h = int((input_h + 2 * this->padding - this->kernel_size) / this->stride + 1);

            this->input = input_tensor; //保存输入张量，用作反向传播时求梯度

            xarray<float> input_pad_tensor = pad(input_tensor, { {0,0},{0,0},{unsigned long long(this->padding),unsigned long long(this->padding)},
                                                                 {unsigned long long(this->padding),unsigned long long(this->padding)} }, pad_mode::constant, 0);

            xarray<float> output_tensor = zeros<float>({ 1,out_channel,out_w,out_h });

            for (int n = 0; n < out_channel; n++)//输出通道
            {
                for (int w = 0; w < out_w; w++)//输出的长
                {
                    for (int h = 0; h < out_h; h++)//输出的高
                    {

                        //定义当前卷积核对应的开始和结束的位置
                        int w_start = w * this->stride;
                        int w_end = w_start + this->kernel_size;
                        int h_start = h * this->stride;
                        int h_end = h_start + this->kernel_size;

                       /* auto tem = view(this->kernel, all(), n, all(), all()) * view(input_pad_tensor, all(), all(), range(w_start, w_end), range(h_start, h_end));
                        auto tem2 = sum(tem);*/
                        output_tensor(0,n,w,h) = sum(view(this->kernel, all(), n, all(), all()) *
                            view(input_pad_tensor, all(), all(), range(w_start, w_end), range(h_start, h_end)))(0);//对应位置卷积相乘


                    }
                }
            }

            return output_tensor;
        }

        xarray<float> backward(xarray<float>& gradient)
        {

        }
};

int main()
{
    xarray<float> a = { {{1,2},{3,4}}, {{4,6},{7,8}} };
   /* xarray<float> b = { {{1,2},{3,4}}, {{4,6},{7,8}} };*/


    /*auto c=sum(view(a,all(),range(1,2),all())*view(b, all(), range(1, 2), all()));
    auto d = c(0);
    std:: cout << d;*/


    Conv_2d con(3, 3, 3,1,1);
    Conv_2d con2(3, 3, 3,1,1 );

    xarray<float> test = random::randn<float>({ 1,3,500,500 });

    auto out = con.forward(test);
    auto out2 = con2.forward(out);

    auto shape = con2.input.shape();
    
    std::cout << out;
    std::cout << "cmgjbxs!!";
    return 0;
    
}

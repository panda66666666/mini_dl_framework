#ifndef POOLING
#define POOLING
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

#define AVGPOOL2D_NODE_FLOAT32 __AVGPOOL2D_NODE__<FLOAT32>

template <typename T>
class __AVGPOOL2D_NODE__ : public __CAL_NODE__<T>
{
public:
	UNSIGNED_INT32 kernel_size;
	UNSIGNED_INT32 stride;
	UNSIGNED_INT32 padding;

	__AVGPOOL2D_NODE__(UNSIGNED_INT32 kernel_size, UNSIGNED_INT32 stride = 1, UNSIGNED_INT32 padding = 0)
	{
		
		this->kernel_size = kernel_size;
		this->stride = stride;
		this->padding = padding;

		auto useless = 0;
	}

	void AddPreNode(__DATA_NODE__<T>* pre_node)
	{
		try
		{
			if (this->pre_node.size() >= 1)
				throw "The \"pre_node\" of \"__AVGPOOL2D_NODE__\" has been overflowed!!";
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
				throw "The value of length of the \"__AVGPOOL2D_NODE__\" need to be 2,but the actual value of it is not enough!";
		}
		catch (const char* s)
		{
			std::cout << s << '\n';
		}

		try
		{
			if (this->back_node == nullptr)
				throw "The \"back_node\" of \"__AVGPOOL2D_NODE__\" is nullptr!";
		}
		catch (const char* s)
		{
			std::cout << s << '\n';
		}

		try
		{
			// 正向传播
			auto input = this->pre_node[0]->data;

			UNSIGNED_INT32 input_w = this->pre_node[0]->data.shape(2);
			UNSIGNED_INT32 input_h = this->pre_node[0]->data.shape(3);

			UNSIGNED_INT32 out_w = (UNSIGNED_INT32)((input_w + 2 * this->padding - this->kernel_size) / this->stride + 1);
			UNSIGNED_INT32 out_h = (UNSIGNED_INT32)((input_h + 2 * this->padding - this->kernel_size) / this->stride + 1);

			xarray<T> input_pad = pad(this->pre_node[0]->data, { {0, 0}, {0, 0}, {this->padding, this->padding}, {this->padding, this->padding} }, pad_mode::constant, 0);

			xarray<T> output = zeros<T>({ (UNSIGNED_INT32)this->pre_node[0]->data.shape(0), (UNSIGNED_INT32)this->pre_node[0]->data.shape(1), out_w, out_h });

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

					view(output, all(), all(), w, h) = mean((view(input, all(), all(), range(w_start, w_end), range(h_start, h_end))), { 2,3 });


					auto useless = 0;
				}


			this->back_node->data = output;
		/*	 std::cout << this->back_node->data;*/

			this->if_forward.clear();
		}
		catch (std::exception& e)
		{

			std::cout << "__AVGPOOL2D_NODE__Forward:" << e.what() << std::endl;
		}
	}

	void Backward()
	{
		try
		{
			// 反向传播
			 auto output_grad = this->back_node->grad;

			UNSIGNED_INT32 input_w = this->pre_node[0]->data.shape(2);
			UNSIGNED_INT32 input_h = this->pre_node[0]->data.shape(3);

			UNSIGNED_INT32 out_w = (UNSIGNED_INT32)((input_w + 2 * this->padding - this->kernel_size) / this->stride + 1);
			UNSIGNED_INT32 out_h = (UNSIGNED_INT32)((input_h + 2 * this->padding - this->kernel_size) / this->stride + 1);

			xarray<T> input_grad = pad(zeros<T>(this->pre_node[0]->data.shape()), { {0, 0}, {0, 0}, {this->padding, this->padding}, {this->padding, this->padding} }, pad_mode::constant, 0);

			xarray<T> input_pad = pad(this->pre_node[0]->data, { {0, 0}, {0, 0}, {this->padding, this->padding}, {this->padding, this->padding} }, pad_mode::constant, 0);

			for (UNSIGNED_INT32 w = 0; w < out_w; w++)
				for (UNSIGNED_INT32 h = 0; h < out_h; h++)
				{
					// 定义当前卷积核对应的开始和结束的位置
					UNSIGNED_INT32 w_start = w * this->stride;
					UNSIGNED_INT32 w_end = w_start + this->kernel_size;
					UNSIGNED_INT32 h_start = h * this->stride;
					UNSIGNED_INT32 h_end = h_start + this->kernel_size;

					view(input_grad, all(), all(), range(w_start, w_end), range(h_start, h_end)) += view(output_grad, all(), all(), w, h) / (this->kernel_size * this->kernel_size);

				}


			this->pre_node[0]->grad += view(input_grad, all(), all(),
				range(this->padding, input_w + this->padding),
				range(this->padding, input_h + this->padding));

		

			/* std::cout << this->pre_node[0]->grad;*/
			auto useless = 0;
		}
		catch (std::exception& e)
		{
			std::cout << "__AVGPOOL2D_NODE__Backward:" << e.what() << std::endl;
		}
	}
};

#endif
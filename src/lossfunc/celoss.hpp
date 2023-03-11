#ifndef CELOSS
#define CELOSS
#include "../datatype.h"
#include "../node/node.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xslice.hpp>
#include <xtensor/xadapt.hpp>
#include <exception>
#include <vector>
#include <queue>
#include <string>
using namespace xt;

#define CELOSS_NODE_FLOAT32 __CELOSS__<FLOAT32>

template <typename T>
class __CELOSS__ : public __CAL_NODE__<T>
{
public:
    void AddPreNode(__DATA_NODE__<T> *pre_node)
    {
        try
        {
            if (this->pre_node.size() >= 2)
                throw "The \"pre_node\" of \"__CE_NODE__\" has been overflowed!!";
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
                throw "The value of length of the \"__CE_NODE__\" need to be 2,but the actual value of it is not enough!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            if (this->back_node == nullptr)
                throw "The \"baak_node\" of \"__CE_NODE__\" is nullptr!";
        }
        catch (const char *s)
        {
            std::cout << s << '\n';
        }

        try
        {
            // this->back_node->data = mean(pow(this->pre_node[0]->data - this->pre_node[1]->data, 2));
            auto pre1 = this->pre_node[0]->data;
            auto pre2 = this->pre_node[1]->data;

            auto dim = pre1.dimension();

            // if (dim == 1)
            // {
            //     pre1 = view(pre1, newaxis(), all());
            //     pre2 = view(pre2, newaxis(), all());
            // }

            auto sum_epow = view(sum(exp(pre1), 1), all(), newaxis());

            this->back_node->data = view(-sum(pre2 * log(exp(pre1) / sum_epow), 1), xt::all(), xt::newaxis());

            this->back_node->data = mean(this->back_node->data, 0);

            // if (dim == 1)
            // {
            //     this->back_node->data = view(this->back_node->data, 3, all());
            // }

            this->if_forward.clear();
        }
        catch (std::exception &e)
        {
            std::cout << "__CE_NODE__Forward:" << e.what() << std::endl;
        }
    }

    void Backward()
    {
        try
        {

            auto pre1 = this->pre_node[0]->data;
            auto pre2 = this->pre_node[1]->data;

            this->pre_node[0]->grad = zeros<T>(pre1.shape());
            this->pre_node[1]->grad = zeros<T>(pre2.shape());

            auto sum_epow = sum(exp(pre1), 1);

            for (UNSIGNED_INT32 i = 0; i < pre1.shape(1); i++)
            {

                auto element1 = -exp(view(pre1, all(), i)) *

                                ((sum_epow - exp(view(pre1, all(), i))) / pow(sum_epow, 2)) *

                                (sum_epow / exp(view(pre1, all(), i))) *

                                view(pre2, all(), i);

                xarray<T> element2 = zeros<T>(element1.shape());

                for (UNSIGNED_INT32 j = 0; j < pre1.shape(1); j++)
                {
                    if (j == i)
                        continue;
                    else
                    {

                        element2 += exp(view(pre1, all(), i)) *

                                    (exp(view(pre1, all(), j)) / pow(sum_epow, 2)) *

                                    (sum_epow / exp(view(pre1, all(), j))) *

                                    view(pre2, all(), j);
                    }
                }

                view(this->pre_node[0]->grad, all(), i) = (element1 + element2);
            }

            for (UNSIGNED_INT32 i = 0; i < pre2.shape(1); i++)
            {
                view(this->pre_node[1]->grad, all(), i) = -log(exp(view(pre1, all(), i)) / sum_epow);
            }

            this->pre_node[0]->grad /= pre1.shape(0);
            this->pre_node[1]->grad /= pre2.shape(0);

            // std::cout << this->pre_node[0]->grad;
            // auto useless = 1;
        }
        catch (std::exception &e)
        {
            std::cout << "__CE_NODE__Backward:" << e.what() << std::endl;
        }
    }
};

#endif
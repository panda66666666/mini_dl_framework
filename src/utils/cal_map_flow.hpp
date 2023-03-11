#ifndef CAL_MAP_FLOW
#define CAL_MAP_FLOW

#include "../datatype.h"
#include "../node/node.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <exception>
#include <vector>
#include <queue>
#include <string>
#include <set>
using namespace xt;

#define CAL_MAP_FLOAT32 __CAL_MAP__<FLOAT32>

template <typename T>
class __CAL_MAP__
{
public:
    std::vector<__DATA_NODE__<T> *> begin_node_list;
    std::vector<__DATA_NODE__<T> *> end_node_list;
    std::vector<__DATA_NODE__<T> *> data_node_list;
    std::vector<__DATA_NODE__<T> *> para_node_list;

    __CAL_MAP__() {}

    __CAL_MAP__(__DATA_NODE__<T> *begin_node, __DATA_NODE__<T> *end_node)
    {
        this->begin_node_list.push_back(begin_node);
        this->end_node_list.push_back(end_node);
    }

    __CAL_MAP__(std::vector<__DATA_NODE__<T> *> &begin_node_list, std::vector<__DATA_NODE__<T> *> &end_node_list)
    {
        this->begin_node_list = begin_node_list;
        this->end_node_list = end_node_list;
    }

    void AddBeginNode(__DATA_NODE__<T> *begin_node)
    {
        this->begin_node_list.push_back(begin_node);
    }

    void AddEndNode(__DATA_NODE__<T> *end_node)
    {
        this->end_node_list.push_back(end_node);
    }

    void AddDataNode(__DATA_NODE__<T> *data_node)
    {
        this->data_node_list.push_back(data_node);
    }

    void AddParaNode(__DATA_NODE__<T> *para_node)
    {
        this->para_node_list.push_back(para_node);
    }

    void Forward()
    {
        try
        {
            if (this->begin_node_list.empty() || this->end_node_list.empty())
                throw "Both the \"begin_node_list\" and \"end_node_list\" of the \"__CAL_MAP__\" can`t be empty!";
        }
        catch (const char *s)
        {
            std::cout << s << std::endl;
        }

        std::queue<__DATA_NODE__<T> *> q_data;
        std::queue<__CAL_NODE__<T> *> q_cal;

        // 迭代器不能用模板定义！！！
        for (auto iter = this->begin_node_list.begin(); iter != this->begin_node_list.end(); iter++)
            q_data.push(*iter);

        while (!q_data.empty() || !q_cal.empty())
        {
            if (!q_data.empty())
            {
                auto tem_node = q_data.front();
                q_data.pop();

                if (tem_node->back_node.size() == 0)
                    continue;
                else
                {
                    for (auto iter = tem_node->back_node.begin(); iter != tem_node->back_node.end(); iter++)
                    {
                        (*iter)->if_forward.insert(tem_node);

                        std::set<__DATA_NODE__<T> *> s;
                        for (auto pre_node = (*iter)->pre_node.begin(); pre_node != (*iter)->pre_node.end(); pre_node++)
                            s.insert(*pre_node);

                        if ((*iter)->if_forward.size() == s.size())
                        {
                            q_cal.push(*iter);
                        }
                    }
                }
            }
            else
            {
                auto tem_node = q_cal.front();
                q_cal.pop();

                tem_node->Forward();
                q_data.push(tem_node->back_node);
            }
        }
    }

    void Backward()
    {
        try
        {
            if (this->begin_node_list.empty() || this->end_node_list.empty())
                throw "Both the \"begin_node_list\" and \"end_node_list\" of the \"__CAL_MAP__\" can`t be empty!";
        }
        catch (const char *s)
        {
            std::cout << s << std::endl;
        }

        std::queue<__DATA_NODE__<T> *> q_data;
        std::queue<__CAL_NODE__<T> *> q_cal;

        for (auto iter = this->end_node_list.begin(); iter != this->end_node_list.end(); iter++)
            q_data.push(*iter);

        while (!q_data.empty() || !q_cal.empty())
        {
            if (!q_data.empty())
            {
                auto tem_node = q_data.front();
                q_data.pop();

                tem_node->Backward();
                // std::cout << tem_node->grad << endl;
                if (tem_node->pre_node == nullptr)
                    continue;
                else
                    q_cal.push(tem_node->pre_node);
            }
            else
            {
                auto tem_node = q_cal.front();
                q_cal.pop();

                tem_node->Backward();

                for (auto iter = tem_node->pre_node.begin(); iter != tem_node->pre_node.end(); iter++)
                {
                    (*iter)->if_backward.insert(tem_node);

                    if ((*iter)->if_backward.size() == (*iter)->back_node.size())
                    {
                        q_data.push(*iter);
                    }
                }
            }
        }
    }

    void SetGradZero()
    {
        // std::cout << this->data_node_list.size();
        // for (int n = 0; n < this->data_node_list.size(); n++)
        //     {

        //     std::cout << n << std::endl;

        //     }

        for (auto iter = this->data_node_list.begin(); iter != this->data_node_list.end(); iter++)
        {
            (*iter)->grad = (T)0;

            if ((*iter)->pre_node != nullptr)
                (*iter)->pre_node->ZeroGrad();
        }
    }

    void UpdatPara(T eta)
    {
        for (auto iter = this->para_node_list.begin(); iter != this->para_node_list.end(); iter++)
        {
            (*iter)->data -= eta * (*iter)->grad;

            if ((*iter)->pre_node != nullptr)
                (*iter)->pre_node->UpdatePara(eta);
        }

        for (auto iter = this->data_node_list.begin(); iter != this->data_node_list.end(); iter++)
        {
           if ((*iter)->pre_node != nullptr)
                (*iter)->pre_node->UpdatePara(eta);
        }

    }
};

#endif
#ifndef MATMUL
#define MATMUL
#include <stddef.h>
#include <typeinfo>
#include <stdexcept>

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>


template <class _Tp>
xt::xarray<_Tp> matmul(xt::xarray<_Tp> &a, xt::xarray<_Tp> &b) noexcept(false)
{
    if (a.shape().size() != b.shape().size()) {
        throw std::runtime_error("Shape mismatching!");
    }

    if (typeid(_Tp).hash_code() != typeid(int).hash_code()
        && typeid(_Tp).hash_code() != typeid(size_t).hash_code()
        && typeid(_Tp).hash_code() != typeid(float).hash_code()
        && typeid(_Tp).hash_code() != typeid(double).hash_code()) {
        throw std::runtime_error("Element type mismatching!");
    }

    xt::xarray<double>::shape_type shape = a.shape();
    *(shape.end()-1) = *(b.shape().end()-1);
    xt::xarray<_Tp> out(shape);

    // std::cout << out;
    //
    // Both argument are 2-D, end the recursion.
    //
    //      a - (M, N)
    //      b - (M, N)
    // 
    if (shape.size() == 2) {
        xt::xarray<_Tp> matrix, vector;
        for (int col = 0; col < *(b.shape().end() - 1); col++) {
            matrix = xt::view(a, xt::all(), xt::all());
            vector = xt::view(b, xt::all(), col);

            // std::cout << "matrix: " << matrix << std::endl
            //           << "vector: " << vector << std::endl;
            xt::view(out, xt::all(), col) = xt::linalg::dot(matrix, vector);
        }
    }
    //
    // Both arguments are N-D, go into deeper layer.
    //
    //
    //      a - (..., M, N)
    //      b - (..., M, N)
    //
    else {
        xt::xarray<_Tp> aSlice, bSlice;
        auto layer = shape.size();
        for (int i = 0; i < shape[0]; i++) {
            aSlice = xt::view(a, i);
            bSlice = xt::view(b, i);
            xt::view(out, i) = matmul(aSlice, bSlice);
        }
    }

    return out;
}

#endif

#pragma once
#include <cmath>
#include "matrix.hpp"

inline float sigmoid_forward(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

inline float loss_func_sse(const Matrix &output, const Matrix &train) // Sum of Squared Errors (SSE)
{
    size_t ro = output.get_rows();
    size_t co = output.get_cols();

    if (ro != train.get_rows() || co != train.get_cols())
    {
        throw std::invalid_argument("Not comparable\n");
    }

    float res = 0.0f;
    for (size_t i = 0; i < ro; ++i)
    {
        for (size_t j = 0; j < co; ++j)
        {
            float diff = output(i, j) - train(i, j);
            res += diff * diff;
        }
    }

    return res;
}
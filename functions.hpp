#pragma once
#include <cmath>

inline float sigmoid_forward(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}
#pragma once
#include <vector>

namespace vicetriceNN
{
    struct Pixel
    {
        unsigned char r, g, b;
    };

    inline float relu(float x) { return x > 0 ? x : 0; }
    inline float relu_derivative(float x)
    {
        return x > 0.0f ? 1.0f : 0.0f;
    }
    std::vector<float> softmax(const std::vector<float> &x);
}

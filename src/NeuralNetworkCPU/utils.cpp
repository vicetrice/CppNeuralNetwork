#include "utils.hpp"

#include <algorithm>
#include <cmath>

namespace vicetriceNN
{
    std::vector<float> softmax(const std::vector<float> &x)
    {
        std::vector<float> result(x.size());

        float max_val = *std::max_element(x.begin(), x.end());

        float sum = 0.0f;
        for (size_t i = 0; i < x.size(); i++)
        {
            result[i] = std::exp(x[i] - max_val); 
            sum += result[i];
        }

        for (size_t i = 0; i < x.size(); i++)
            result[i] /= sum;

        return result;
    }

} // namespace vicetriceNN

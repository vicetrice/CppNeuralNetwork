#include "utils.hpp"

#include <algorithm>
#include <cmath>

namespace vicetriceNN
{
    void softmax(std::vector<float> &x)
    {

        float max_val = *std::max_element(x.begin(), x.end());

        float sum = 0.0f;
        for (size_t i = 0; i < x.size(); i++)
        {
            x[i] = std::exp(x[i] - max_val);
            sum += x[i];
        }

        for (size_t i = 0; i < x.size(); i++)
            x[i] /= sum;
    }

} // namespace vicetriceNN

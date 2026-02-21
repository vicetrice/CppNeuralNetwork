#include "neuronLayer.hpp"
#include "utils.hpp"
#include <random>

namespace vicetriceNN
{

    neuronLayer::neuronLayer(int in_size, int out_size)
        : input_size(in_size), output_size(out_size)
    {
        weights.resize(out_size, std::vector<float>(in_size));
        biases.resize(out_size, 0.0f);

        // Random ini. Could be specific values if you know the weights beforehand. Maybe change it later?
        std::mt19937 gen(0);
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        for (int i = 0; i < out_size; i++)
        {
            for (int j = 0; j < in_size; j++)
            {
                weights[i][j] = dist(gen);
            }
        }
    }

    neuronLayer::~neuronLayer() = default;

    std::vector<float> neuronLayer::forward(const std::vector<float> &input) const
    {
        std::vector<float> output(output_size, 0.0f);
        for (int i = 0; i < output_size; i++)
        {
            for (int j = 0; j < input_size; j++)
            {
                output[i] += weights[i][j] * input[j];
            }

            output[i] += biases[i];
            output[i] = relu(output[i]);
        }
        return output;
    }

 
}

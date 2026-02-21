#pragma once
#include <vector>
#include <string>
#include "neuronLayer.hpp"
#include "mnist_images.hpp"

namespace vicetriceNN
{
    class neuralNetwork
    {
    private:
        std::vector<neuronLayer> layers;
        float learning_rate = 0.01f;

        static float cross_entropy_loss(const std::vector<float> &output, int label);

    public:
        neuralNetwork() = default;
        ~neuralNetwork() = default;

        void addLayer(int input_size, int output_size);

        std::vector<float> forward(const std::vector<float> &input,
                                   std::vector<std::vector<float>> &layer_outputs) const;

        void backward(const std::vector<float> &target, const std::vector<std::vector<float>> &layer_outputs);

        void setLearningRate(float lr) { learning_rate = lr; }

        void train(const mnist_images &dataset, int epochs, int batch_size);

        float evaluate(const mnist_images &dataset) const;

        std::vector<float> predict(const std::vector<float> &input) const;

        void saveWeights(const std::string &filename) const;
        bool loadWeights(const std::string &filename);

        inline size_t getNumLayers() const { return layers.size(); }
        inline const neuronLayer &getLayer(size_t idx) const { return layers[idx]; }
    };
}

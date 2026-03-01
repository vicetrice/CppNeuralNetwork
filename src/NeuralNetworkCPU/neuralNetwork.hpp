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
        float learning_rate = 0.001f;
        float lambda = 0.0f;
        float loss;

        static float cross_entropy_loss(const std::vector<float> &output, int label);
        void computeDelta(const std::vector<float> &output, const std::vector<float> &target, std::vector<float> &delta);
        void computeNewDelta(const std::vector<float> &prev_output, const std::vector<float> &old_delta, const neuronLayer &layer, std::vector<float> &new_delta);
        void computeWeights(neuronLayer &layer, const std::vector<float> &delta, const std::vector<float> &prev_output);

    public:
        neuralNetwork() = default;
        ~neuralNetwork() = default;

        void addLayer(int input_size, int output_size);

        std::vector<float> forward(const std::vector<float> &input,
                                   std::vector<std::vector<float>> &layer_outputs) const;

        void backward(const std::vector<float> &target, const std::vector<std::vector<float>> &layer_outputs);

        inline void setLearningRate(float lr) { learning_rate = lr; }
        inline void setLambda(float lmbd) { lambda = lmbd; }

        void train(const mnist_images &dataset, int epochs, int batch_size);

        float evaluate(const mnist_images &dataset) const;

        std::vector<float> predict(const std::vector<float> &input) const;

        void saveWeights(const std::string &filename) const;
        bool loadWeights(const std::string &filename);

        void saveFullModel(const std::string &filename) const;
        bool loadFullModel(const std::string &filename);

        inline size_t getNumLayers() const { return layers.size(); }
        inline const neuronLayer &getLayer(size_t idx) const { return layers[idx]; }

        inline float getLoss() const { return loss; }
        inline float getLearningRate() const { return learning_rate; }
        inline float getLambda() const { return lambda; }
    };
}

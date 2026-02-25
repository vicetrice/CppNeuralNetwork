#include <iostream>
#include "mnist_images.hpp"
#include "neuralNetworkGPU.hpp"

int main()
{
    vicetriceNN::mnist_images train_dataset;
    if (!train_dataset.load("../archive/train-images.idx3-ubyte",
                            "../archive/train-labels.idx1-ubyte"))
    {
        std::cerr << "Error cargando MNIST train\n";
        return 1;
    }

    int input_size = train_dataset.getRows() * train_dataset.getCols();
    int hidden1 = 128;
    int hidden2 = 64;
    int output_size = 10;
    float learning_rate = 0.01f;

    vicetriceNN::neuralNetworkGPU net;
    net.addLayer(input_size, hidden1);
    net.addLayer(hidden1, hidden2);
    net.addLayer(hidden2, output_size);
    net.setLearningRate(learning_rate);

    int epochs = 6;
    int batch_size = 32;

    std::cout << "Entrenando la red GPU...\n";
    net.train(train_dataset, epochs, batch_size);
    std::cout << "Entrenamiento completado.\n";

    vicetriceNN::mnist_images test_dataset;
    if (!test_dataset.load("../archive/t10k-images.idx3-ubyte",
                           "../archive/t10k-labels.idx1-ubyte"))
    {
        std::cerr << "Error cargando MNIST test\n";
        return 1;
    }

    float test_accuracy = net.evaluate(test_dataset);
    std::cout << "Precision en el conjunto de test (10k imagenes): "
              << test_accuracy * 100.0f << "%\n";

    return 0;
}
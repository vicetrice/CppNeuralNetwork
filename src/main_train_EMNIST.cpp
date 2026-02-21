#include <iostream>
#include "mnist_images.hpp"
#include "neuralNetwork.hpp"

int main()
{
    using namespace vicetriceNN;

    mnist_images train_dataset;

    if (!train_dataset.load(
            "EMNIST_archive/emnist-balanced-train-images-idx3-ubyte",
            "EMNIST_archive/emnist-balanced-train-labels-idx1-ubyte"))
    {
        std::cerr << "Error cargando EMNIST Balanced train\n";
        return 1;
    }

    mnist_images test_dataset;

    if (!test_dataset.load(
            "EMNIST_archive/emnist-balanced-test-images-idx3-ubyte",
            "EMNIST_archive/emnist-balanced-test-labels-idx1-ubyte"))
    {
        std::cerr << "Error cargando EMNIST Balanced test\n";
        return 1;
    }

    int input_size = train_dataset.getRows() * train_dataset.getCols();
    int hidden1 = 256;
    int hidden2 = 128;
    int output_size = 47;

    float learning_rate = 0.001f;

    neuralNetwork net;
    net.addLayer(input_size, hidden1);
    net.addLayer(hidden1, hidden2);
    net.addLayer(hidden2, output_size);
    net.setLearningRate(learning_rate);

    if (!net.loadWeights("weights_EMNIST/emnist_balanced_weights.bin"))
    {
        std::cout << "No se encontraron pesos guardados\n";
        std::cout << "Entrenando desde cero\n";

        int epochs = 1;
        int batch_size = 64;

        net.train(train_dataset, epochs, batch_size);

        net.saveWeights("emnist_balanced_weights.bin");
        std::cout << "Pesos guardados en emnist_balanced_weights.bin\n";
    }
    else
    {
        std::cout << "Pesos cargados correctamente\n";

        char choice;
        std::cout << "Desea continuar entrenando? (s/n): ";
        std::cin >> choice;

        if (choice == 's' || choice == 'S')
        {
            int epochs, batch_size;

            std::cout << "Ingrese epochs adicionales: ";
            std::cin >> epochs;

            std::cout << "Ingrese batch size: ";
            std::cin >> batch_size;

            net.train(train_dataset, epochs, batch_size);
            net.saveWeights("emnist_balanced_weights.bin");

            std::cout << "Pesos actualizados guardados\n";
        }
    }

    std::cout << "\nEvaluando en test set\n";

    float accuracy = net.evaluate(test_dataset);

    std::cout << "Precision en EMNIST Balanced: "
              << accuracy * 100.0f << "%\n";

    return 0;
}

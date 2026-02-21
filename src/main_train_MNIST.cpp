#include <iostream>
#include <algorithm>
#include <iomanip>
#include <vector>
#include "mnist_images.hpp"
#include "neuralNetwork.hpp"

int main()
{
    vicetriceNN::mnist_images dataset;
    if (!dataset.load("archive/train-images.idx3-ubyte",
                      "archive/train-labels.idx1-ubyte"))
    {
        std::cerr << "Error cargando MNIST\n";
        return 1;
    }

    int input_size = dataset.getRows() * dataset.getCols();
    int hidden1 = 64 * 2;
    int hidden2 = 32 * 2;
    int output_size = 10;

    float learning_rate = 0.01f;

    vicetriceNN::neuralNetwork net;
    net.addLayer(input_size, hidden1);
    net.addLayer(hidden1, hidden2);
    net.addLayer(hidden2, output_size);
    net.setLearningRate(learning_rate);

    if (!net.loadWeights("weights.bin"))
    {
        std::cout << "No se encontraron pesos guardados.\n";
        std::cout << "Entrenando desde cero...\n";

        int epochs = 5;
        int batch_size = 32;

        net.train(dataset, epochs, batch_size);

        net.saveWeights("weights.bin");
        std::cout << "Pesos guardados en weights.bin\n";
    }
    else
    {
        std::cout << "Pesos cargados desde weights.bin\n";

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

            net.train(dataset, epochs, batch_size);
            net.saveWeights("weights.bin");

            std::cout << "Pesos actualizados guardados.\n";
        }
    }

    for (;;)
    {
        std::cout << "\nIngrese el indice de la imagen a predecir (0-"
                  << dataset.size() - 1
                  << ") o -1 para salir: ";

        int idx;
        std::cin >> idx;

        if (idx == -1)
        {
            std::cout << "Saliendo...\n";
            break;
        }

        if (idx < 0 || static_cast<size_t>(idx) >= dataset.size())
        {
            std::cerr << "indice fuera de rango. Intente de nuevo.\n";
            continue;
        }

        const auto &img = dataset.getImage(idx);

        std::vector<float> input(input_size);

        int rows = dataset.getRows();
        int cols = dataset.getCols();

        for (int i = 0; i < input_size; i++)
        {
            input[i] = img[i].r / 255.0f;
        }

        std::cout << "\nMatriz de entrada (valores >0 en verde):\n\n";
        std::cout << std::fixed << std::setprecision(2);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                int index = r * cols + c;
                float value = input[index];

                if (value == 0.0f)
                    std::cout << "\033[90m" << std::setw(5) << value << "\033[0m ";
                else
                    std::cout << "\033[32m" << std::setw(5) << value << "\033[0m ";
            }
            std::cout << "\n";
        }

        auto output_probs = net.predict(input);

        std::cout << "\nProbabilidades:\n";

        size_t predicted = 0;
        float max_value = output_probs[0];
        for (size_t i = 0; i < output_probs.size(); i++)
        {
            std::cout << "Digito " << i << " -> " << output_probs[i] << "\n";
            if (output_probs[i] > max_value)
            {
                max_value = output_probs[i];
                predicted = i;
            }
        }

        std::cout << "\nEtiqueta real: "
                  << static_cast<int>(dataset.getLabel(idx))
                  << "\nPrediccion: "
                  << predicted << "\n";
    }

    return 0;
}

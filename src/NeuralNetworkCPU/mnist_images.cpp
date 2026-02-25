#include "mnist_images.hpp"

namespace vicetriceNN
{
    
    mnist_images::mnist_images() = default;
    mnist_images::~mnist_images() = default;

    // ---------- PRIVATE METHODS -----------------------
  
    bool mnist_images::loadImages(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
            return false;

        uint32_t magic = 0;
        file.read(reinterpret_cast<char *>(&magic), 4);
        magic = swapEndian(magic);
        if (magic != 2051)
            return false;

        uint32_t n_images = 0, n_rows = 0, n_cols = 0;
        file.read(reinterpret_cast<char *>(&n_images), 4);
        file.read(reinterpret_cast<char *>(&n_rows), 4);
        file.read(reinterpret_cast<char *>(&n_cols), 4);

        n_images = swapEndian(n_images);
        rows = swapEndian(n_rows);
        cols = swapEndian(n_cols);

        images.resize(n_images, std::vector<Pixel>(rows * cols));

        for (uint32_t i = 0; i < n_images; ++i)
        {
            for (int j = 0; j < rows * cols; ++j)
            {
                unsigned char p = 0;
                file.read(reinterpret_cast<char *>(&p), 1);
                images[i][j] = {p, p, p};
            }
        }
        file.close();
        return true;
    }

    bool mnist_images::loadLabels(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
            return false;

        uint32_t magic = 0;
        file.read(reinterpret_cast<char *>(&magic), 4);
        magic = swapEndian(magic);
        if (magic != 2049)
            return false;

        uint32_t n_labels = 0;
        file.read(reinterpret_cast<char *>(&n_labels), 4);
        n_labels = swapEndian(n_labels);

        labels.resize(n_labels);
        file.read(reinterpret_cast<char *>(labels.data()), n_labels);
        file.close();
        return true;
    }



    // ----------------- PUBLIC METHODS  ----------------------------------------
    bool mnist_images::load(const std::string &images_file, const std::string &labels_file)
    {
        if (!loadImages(images_file))
            return false;
        if (!loadLabels(labels_file))
            return false;
        if (images.size() != labels.size())
            return false;
        return true;
    }

   


    bool mnist_images::saveBMP(const std::string &filename, size_t idx) const
    {
        if (idx >= images.size())
            return false;

        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open())
            return false;

        unsigned char header[54] = {
            0x42, 0x4D, 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0,
            40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        int row_padded = (cols * 3 + 3) & (~3);
        int filesize = 54 + row_padded * rows;
        header[2] = filesize;
        header[3] = filesize >> 8;
        header[4] = filesize >> 16;
        header[5] = filesize >> 24;

        header[18] = cols;
        header[19] = cols >> 8;
        header[20] = cols >> 16;
        header[21] = cols >> 24;

        header[22] = rows;
        header[23] = rows >> 8;
        header[24] = rows >> 16;
        header[25] = rows >> 24;

        file.write(reinterpret_cast<char *>(header), 54);

        std::vector<unsigned char> row(row_padded);
        for (int y = rows - 1; y >= 0; y--)
        {
            for (int x = 0; x < cols; x++)
            {
                const Pixel &p = images[idx][y * cols + x];
                row[x * 3 + 0] = p.b;
                row[x * 3 + 1] = p.g;
                row[x * 3 + 2] = p.r;
            }
            file.write(reinterpret_cast<char *>(row.data()), row_padded);
        }
        file.close();
        return true;
    }
}

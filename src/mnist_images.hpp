#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include "utils.hpp"

namespace vicetriceNN
{
    
    class mnist_images
    {
    private:
        std::vector<std::vector<Pixel>> images;
        std::vector<unsigned char> labels;
        int rows = 0, cols = 0;

        static inline uint32_t swapEndian(uint32_t val)
        {
            return __builtin_bswap32(val);
        }

        bool loadImages(const std::string &filename);
        bool loadLabels(const std::string &filename);

    public:
        mnist_images();
        ~mnist_images();

        bool load(const std::string &images_file, const std::string &labels_file);

        inline int getRows() const { return rows; }
        inline int getCols() const { return cols; }
        inline size_t size() const { return images.size(); }

        const inline std::vector<Pixel> &getImage(size_t idx) const { return images[idx]; }
        unsigned char inline getLabel(size_t idx) const { return labels[idx]; }

        bool saveBMP(const std::string &filename, size_t idx) const;
    };
}

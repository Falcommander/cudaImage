#pragma once
#include <string>
void grayscaleStains(std::string file);
void grayscaleCase(std::string file, size_t casePerLine = 7);
void grayscaleWithoutOne(std::string file);
void colored_sobel(std::string file);
void convolution_matrix(std::string file);
void launch();
void choiceGrayscaleCase(const std::string file);
inline bool file_exist(const std::string& file);
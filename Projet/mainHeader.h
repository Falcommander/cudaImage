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
void andyWarhol(const int duplicationNumber = 4);

void grayscaleStainsCPU();
void grayscaleCaseCPU(const size_t casePerLine = 7);
void grayscaleWithoutOneCPU();
void andyWarholCPU(const int duplicationNumber = 4);
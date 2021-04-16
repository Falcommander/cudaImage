CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
PROJECT("PROJET")

SET(CMAKE_CXX_STANDARD 20)
SET(CMAKE_C_STANDARD 20)
ENABLE_LANGUAGE(CUDA)

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

INCLUDE_DIRECTORIES(include)
FILE(GLOB COMMON
 andyWarhol.cpp andyWarhol.cu
 coloredSobel.cu coloredSobel.cpp
 convolution_matrix.cu convolution_matrix.cpp
 grayscaleCase.cpp grayscaleCase.cu
 grayscaleStains.cpp grayscaleStains.cu
 grayscaleWithoutOne.cpp grayscaleWithoutOne.cu
 launcher.cpp
 utils.cu
)
ADD_EXECUTABLE(launcher ${COMMON})
TARGET_LINK_LIBRARIES(launcher ${OpenCV_LIBS})
SET(CMAKE_CUDA_FLAGS  "${CMAKE_CXX_FLAGS} -rdc=true")

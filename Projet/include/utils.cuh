#pragma once

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#include <opencv2/opencv.hpp>

using namespace std;

__device__ double cuSquare(double val);
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

using namespace std;

__global__ void filter(unsigned char const* in, unsigned char* const out, std::size_t w, std::size_t h)
{
	auto i = blockIdx.x * blockDim.x + threadIdx.x;
	auto j = blockIdx.y * blockDim.y + threadIdx.y;

	int const hor1 = -2; int const hor2 = -1; int const hor3 = 0;	
	int const hor4 = -1; int const hor5 = 1; int const hor6 = 1;		
	int const hor7 = 0; int const hor8 = 1; int const hor9 = 2;	
	
	
	if (i > 0 && j > 0 && i < 3*w-1 && j < 3*h-1)
	{
		auto hh = hor1 * in[(j - 1) * w + i - 1] + hor2 * in[(j - 1) * w + i] + hor3 * in[(j - 1) * w + i + 1]
			+ hor4 * in[j * w + i - 1] + hor5 * in[j * w + i] +hor6 * in[j * w + i + 1]
			+ hor7 * in[(j + 1) * w + i - 1] + hor8 * in[(j + 1) * w + i] + hor9 * in[(j + 1) * w + i + 1];

		auto vv = hor1 * in[(j - 1) * w + i - 1] + hor2 * in[j * w + i - 1] + hor3 * in[(j + 1) * w + i - 1]
			+ hor4 * in[(j-1) * w + i] + hor5 * in[j * w + i] + hor6 * in[(j+1) * w + i]
			+ hor7 * in[(j - 1) * w + i + 1] + hor8 * in[j * w + i + 1] + hor9* in[(j + 1) * w + i + 1];

		auto res = hh * hh + vv * vv;
		res = res > 255 * 255 ? 255 * 255 : res;
		out[j * w + i] = sqrtf(res);

		/*hh = hor1 * in[2*((j - 1) * w + i) - 1] + hor2 * in[2*((j - 1) * w + i)] + hor3 * in[2*((j - 1) * w + i) + 1]
			+ hor4 * in[2*(j * w + i) - 1] + hor5 * in[2*(j * w + i)] + hor6 * in[2*(j * w + i) + 1]
			+ hor7 * in[2*((j + 1) * w + i) - 1] + hor8 * in[2*((j + 1) * w + i)] + hor9 * in[2*((j + 1) * w + i) + 1];

		vv = hor1 * in[2*((j - 1) * w + i) - 1] + hor2 * in[2*(j * w + i) - 1] + hor3 * in[2*((j - 1) * w + i) - 1]
			+ hor4 * in[2*((j - 1) * w + i)] + hor5 * in[2*(j * w + i)] + hor6 * in[2*((j + 1) * w + i)]
			+ hor7 * in[2*((j - 1) * w + i) + 1] + hor8 * in[2*(j * w + i) + 1] + hor9 * in[2*((j + 1) * w + i) + 1];

		res = hh * hh + vv * vv;
		res = res > 255 * 255 ? 255 * 255 : res;
		out[2*(j * w + i)] = sqrt((float)res);

		auto h = hor1 * in[3 * ((j - 1) * w + i) - 1] + hor2 * in[3 * ((j - 1) * w + i)] + hor3 * in[3 * ((j - 1) * w + i) + 1]
			+ hor4 * in[3 * (j * w + i) - 1] + hor5 * in[3 * (j * w + i)] + hor6 * in[3 * (j * w + i) + 1]
			+ hor7 * in[3 * ((j + 1) * w + i) - 1] + hor8 * in[3 * ((j + 1) * w + i)] + hor9 * in[3 * ((j + 1) * w + i) + 1];

		auto v = hor1 * in[3 * ((j - 1) * w + i) - 1] + hor2 * in[3 * (j * w + i) - 1] + hor3 * in[3 * ((j + 1) * w + i) - 1]
			+ hor4 * in[3 * ((j - 1) * w + i)] + hor5 * in[3 * (j * w + i)] + hor6 * in[3 * ((j + 1) * w + i)]
			+ hor7 * in[3 * ((j - 1) * w + i) + 1] + hor8 * in[3 * (j * w + i) + 1] + hor9 * in[3 * ((j + 1) * w + i) + 1];

		auto res = h * h + v * v;
		res = res > 255 * 255 ? 255 * 255 : res;
		out[(j * w + i)] = sqrtf(res);*/
	}

}

void filter()
{
	cv::Mat m_in = cv::imread("croise.jpg", cv::IMREAD_UNCHANGED);
	auto rgb = m_in.data;
	auto rows = m_in.rows;
	auto cols = m_in.cols;

	std::vector< unsigned char > g(3 * rows * cols);
	cv::Mat m_out(rows, cols, CV_8UC3, g.data());

	unsigned char* rgb_d;
	unsigned char* out_d;

	auto start = std::chrono::system_clock::now();
	cudaEvent_t cudaStart, cudaStop;
	cudaEventCreate(&cudaStart);
	cudaEventCreate(&cudaStop);

	cudaEventRecord(cudaStart);

	cudaMalloc(&rgb_d, 3 * rows * cols);
	cudaMalloc(&out_d, 3 * rows * cols);

	cudaMemcpy(rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice);

	dim3 block(32, 32);
	dim3 grid((cols - 1) / block.x + 1, (rows - 1) / block.y + 1); //(4,4)

	filter << <grid, block >> > (rgb_d, out_d, cols, rows);

	cudaMemcpy(g.data(), out_d, 3 * rows * cols, cudaMemcpyDeviceToHost);

	cudaEventRecord(cudaStop);
	cudaEventSynchronize(cudaStop);
	auto stop = std::chrono::system_clock::now();


	auto duration = stop - start;
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, cudaStart, cudaStop);
	std::cout << "Temps kernel: " << elapsedTime << std::endl;
	cudaEventDestroy(cudaStart);
	cudaEventDestroy(cudaStop);
	auto err = cudaGetLastError();

	std::cout << "Erreur: " << err << std::endl;

	std::cout << ms << " ms" << std::endl;

	cv::imwrite("filter.jpg", m_out);

	cudaFree(rgb_d);
	cudaFree(out_d);
}
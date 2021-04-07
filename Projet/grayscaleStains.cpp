#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

using namespace std;

void grayscaleStainsCPUKernel(unsigned char* rgb, unsigned char* g, const size_t cols, const size_t rows, const int mult)
{
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			g[i * cols + j] = (
				mult * rgb[3 * (i * cols + j)]
				+ mult * rgb[3 * (i * cols + j) + 1]
				+ mult * rgb[3 * (i * cols + j) + 2]
				) / 1024;
		}
	}
}


void grayscaleStainsCPU()
{
	cv::Mat m_in = cv::imread("ecureuil.jpg", cv::IMREAD_UNCHANGED);

	const int mult = 550;

	auto rgb = m_in.data;
	auto rows = m_in.rows;
	auto cols = m_in.cols;

	srand(time(0));
	std::vector< unsigned char > g(rows * cols);
	cv::Mat m_out(rows, cols, CV_8UC1, g.data());

	//Test de nombre aléatoire pour avoir quelque chose de viable
	//int randomNumber = rand() % 1024;
	//cout << "Random number : " << randomNumber << endl;

	cout << "rows : " << rows << endl;
	cout << "cols : " << cols << endl;
	
	auto start = std::chrono::system_clock::now();

	grayscaleStainsCPUKernel(rgb, g.data(), cols, rows, mult);

	auto stop = std::chrono::system_clock::now();

	auto duration = stop - start;
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

	std::cout << ms << " ms" << std::endl;

	cv::imwrite("out.jpg", m_out);
}
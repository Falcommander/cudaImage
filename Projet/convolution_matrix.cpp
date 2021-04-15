#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <string>

using namespace std;

void convolution_matrixCPUKernel(unsigned char const* in, unsigned char* const out, std::size_t w, std::size_t h)
{

	int const hor1 = -2; int const hor2 = -1; int const hor3 = 0;
	int const hor4 = -1; int const hor5 = 1; int const hor6 = 1;
	int const hor7 = 0; int const hor8 = 1; int const hor9 = 2;


	for (size_t i = 0; i < w; i++)
	{
		for (size_t j = 0; j < h; j++)
		{
			if (i > 1 && j > 1 && i < w - 1 && j < h - 1)
			{
				for (int c = 0; c < 3; c++) {

					auto hh = (hor1 * in[((j - 1) * w + i - 1) * 3 + c] + hor2 * in[((j - 1) * w + i) * 3 + c] + hor3 * in[((j - 1) * w + i + 1) * 3 + c]
						+ hor4 * in[(j * w + i - 1) * 3 + c] + hor5 * in[(j * w + i) * 3 + c] + hor6 * in[(j * w + i + 1) * 3 + c]
						+ hor7 * in[((j + 1) * w + i - 1) * 3 + c] + hor8 * in[((j + 1) * w + i) * 3 + c] + hor9 * in[((j + 1) * w + i + 1) * 3 + c]);

					auto vv = (hor1 * in[((j - 1) * w + i - 1) * 3 + c] + hor2 * in[(j * w + i - 1) * 3 + c] + hor3 * in[((j + 1) * w + i - 1) * 3 + c]
						+ hor4 * in[((j - 1) * w + i) * 3 + c] + hor5 * in[(j * w + i) * 3 + c] + hor6 * in[((j + 1) * w + i) * 3 + c]
						+ hor7 * in[((j - 1) * w + i + 1) * 3 + c] + hor8 * in[(j * w + i + 1) * 3 + c] + hor9 * in[((j + 1) * w + i + 1) * 3 + c]);


					auto res = hh * hh + vv * vv;
					res = res > 255 * 255 ? 255 * 255 : res;
					out[(j * w + i) * 3 + c] = sqrt((float)res);
				}
			}
		}
	}
}

void convolution_matrixCPU(std::string name)
{
	cv::Mat m_in = cv::imread(name, cv::IMREAD_UNCHANGED);
	auto rgb = m_in.data;
	auto rows = m_in.rows;
	auto cols = m_in.cols;

	std::vector< unsigned char > g(3 * rows * cols);
	cv::Mat m_out(rows, cols, CV_8UC3, g.data());

	auto start = std::chrono::system_clock::now();
	
	convolution_matrixCPUKernel(rgb, g.data(), cols, rows);	
	auto stop = std::chrono::system_clock::now();


	auto duration = stop - start;
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
	

	std::cout << ms << " ms" << std::endl;

	cv::imwrite("cmCPU.jpg", m_out);
	cout << "Le fichier \"cmCPU.jpg\" a bien ete genere. Toutes nos felicitations !" << endl;
}
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <numeric>
#include <random>

using namespace std;

void duplicateImageWarholCPUKernel(unsigned char* in, unsigned char* out, const size_t cols, const size_t rows, const size_t duplicationNumber)
{
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			int square = sqrt(duplicationNumber);

			//For each line
			for (int k = 0; k < square; k++) {

				auto index = 3 * (i * cols / square + j / square) + 3 * cols * rows / square * k;

				if (out[index] == 0 && out[index + 1] == 0 && out[index + 2] == 0) {
					out[index] = in[3 * (i * cols + j)];
					out[index + 1] = in[3 * (i * cols + j) + 1];
					out[index + 2] = in[3 * (i * cols + j) + 2];
				}
			}
		}
	}
}


void colorizeImageWarholCPUKernel(unsigned char* in, unsigned char* out, const size_t cols, const size_t rows, unsigned int* caseType, const size_t duplicationNumber)
{
	const int effectNumber = 6;

	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			int square = sqrt(duplicationNumber);

			auto index = 3 * (i * cols + j);

			//For each case
			for (int k = 1; k < square + 1; k++) {
				for (int l = 1; l < square + 1; l++) {
					if (j < cols / square * k && i < rows / square * l) {
						if (out[index] == 0 && out[index + 1] == 0 && out[index + 2] == 0) {

							//Without first channel
							if (caseType[(k - 1) * square + l - 1] % effectNumber == 0) {
								//out[index] = in[index];
								out[index + 1] = in[index + 1];
								out[index + 2] = in[index + 2];
							}
							//Without second channel
							else if (caseType[(k - 1) * square + l - 1] % effectNumber == 1) {
								out[index] = in[index];
								//out[index + 1] = in[index + 1];
								out[index + 2] = in[index + 2];
							}
							//Without third channel
							else if (caseType[(k - 1) * square + l - 1] % effectNumber == 2) {
								out[index] = in[index];
								out[index + 1] = in[index + 1];
								//out[index + 2] = in[index + 2];
							}
							//Grayscale
							else if (caseType[(k - 1) * square + l - 1] % effectNumber == 3) {
								out[index] = (307 * in[index] + 604 * in[index + 1] + 113 * in[index + 2]) / 1024;
								out[index + 1] = (307 * in[index] + 604 * in[index + 1] + 113 * in[index + 2]) / 1024;
								out[index + 2] = (307 * in[index] + 604 * in[index + 1] + 113 * in[index + 2]) / 1024;
							}
							//Invert
							else if (caseType[(k - 1) * square + l - 1] % effectNumber == 4) {
								out[index] = in[index] ^ 0x00ffffff;
								out[index + 1] = in[index + 1] ^ 0x00ffffff;
								out[index + 2] = in[index + 2] ^ 0x00ffffff;
							}
							else if (caseType[(k - 1) * square + l - 1] % effectNumber == 5) {
								out[index] = in[index];
								out[index + 1] = in[index + 1];
								out[index + 2] = in[index + 2];
							}
							//If you add effect here, please update the effect number (const int)
						}
					}
				}
			}
		}
	}
}


void andyWarholCPU(const string file, const int duplicationNumber = 4)
{
	srand(time(0));

	cv::Mat m_in = cv::imread(file, cv::IMREAD_UNCHANGED);

	auto rgb = m_in.data;
	auto rows = m_in.rows;
	auto cols = m_in.cols;

	std::vector< unsigned char > duplicated(3 * rows * cols);
	std::vector< unsigned char > out(3 * rows * cols);
	cv::Mat image_duplicated(rows, cols, CV_8UC3, duplicated.data());
	cv::Mat image_out(rows, cols, CV_8UC3, out.data());

	//Generate unique int for each case of the array
	std::vector<unsigned int> caseType(duplicationNumber);
	std::iota(caseType.begin(), caseType.end(), 0);
	std::shuffle(caseType.begin(), caseType.end(), default_random_engine(0));
	//std::for_each(caseType.begin(), caseType.end(), [](unsigned int& o) {cout << o << " "; });
	cout << endl;

	cout << "rows : " << rows << endl;
	cout << "cols : " << cols << endl;

	auto start = std::chrono::system_clock::now();

	duplicateImageWarholCPUKernel(rgb, duplicated.data(), cols, rows, duplicationNumber);
	colorizeImageWarholCPUKernel(duplicated.data(), out.data(), cols, rows, caseType.data(), duplicationNumber);

	auto stop = std::chrono::system_clock::now();

	auto duration = stop - start;
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

	std::cout << ms << " ms" << std::endl;

	cv::imwrite("originalArrayCPU.jpg", image_duplicated);
	cv::imwrite("andyWarholdCPU.jpg", image_out);
	cout << "Les fichiers \"originalArrayCPU.jpg\" et \"andyWarholdCPU.jpg\" a bien ?t? g?n?r?. Toutes nos f?licitations !" << endl;

}
#include "include/mainHeader.h"
#include <iostream>
#include <string>
using namespace std;

int main() {
	
	launch();
	return 0;
}


void launch()
{
	int choice = 0;
	int choiceProc = 0;
	string file;
	cout << "Ce projet a été réalisé par Samir BELFAQUIR et Romain CORBEAU" << endl;
	cout << "Vous pouvez choisir parmi 5 filtres différents :" << endl;
	cout << "1. Grayscale Strains" << endl;
	cout << "2. Colored Sobel" << endl;
	cout << "3. Emboss" << endl;
	cout << "4. Grayscale without one" << endl;
	cout << "5. Grayscale Case" << endl;
	cout << "6. Andy Warhol" << endl;

	do {
		cin >> choice;
	} while (choice < 1 && choice > 5 && !isdigit(choice));

	cout << "Entrez le nom (avec l'extension) de l'image que vous souhaitez utiliser" << endl;
	cout << "Si vous n'entrez rien, une image par defaut sera utilisee" << endl;

	getline(cin, file);
	if (file.length() == 0)
		file = "croise.jpg";

	do {
		getline(cin, file);
		if (file.length() == 0)
		{
			file = "croise.jpg";
			break;
		}			
	} while (!file_exist(file));

	cout << "Souhaitez-vous une version CPU ou la version CUDA ? (1 pour le CPU, 2 pour CUDA)" << endl;

	do {
		cin >> choiceProc;
	} while (choiceProc != 1 && choiceProc != 2 && !isdigit(choiceProc));

	if (choice == 1)
	{
		if (choiceProc == 1)
			grayscaleStainsCPU(file);
		else
			grayscaleStains(file);
	}		
	else if (choice == 2)
			colored_sobel(file);
	else if (choice == 3)
		convolution_matrix(file);
	else if (choice == 4)
	{
		if (choiceProc == 1)
			grayscaleWithoutOneCPU(file);
		else grayscaleWithoutOne(file);
	}		
	else if (choice == 5)
		choiceGrayscaleCase(file, choiceProc);
}


void choiceGrayscaleCase(const string file, const int choiceProc)
{
	size_t rows = 0;
	cout << "En combien de lignes et de colonnes souhaitez-vous que l'image soit decoupee (minimum 3)  ?" << endl;
	
	do {
		cin >> rows;
	} while (rows < 3 && !isdigit(rows));
	if(choiceProc == 1)
		grayscaleCaseCPU(file, rows);
	else
		grayscaleCase(file, rows);

}

inline bool file_exist(const string& file)
{
	struct stat buffer;
	return (stat(file.c_str(), &buffer) == 0);
}
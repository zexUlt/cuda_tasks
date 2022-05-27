#include <string>
#include <iosfwd>

double* LoadFromFile(int& size, std::string filePath);
void SaveToFile(double* AB, int size);
void Print(double* AB, const int& size);
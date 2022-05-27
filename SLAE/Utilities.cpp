#include <iostream>
#include <fstream>

#include "Utilities.hpp"

double* LoadFromFile(int& size, std::string filePath)
{
    std::ifstream in(filePath, std::ios_base::in);

    in >> size;

    auto matrix_ab = new double[size * (size + 1)];

    // Reading matrix A
    for(auto i = 0u; i < size*(size + 1); ++i){
        if( (i + 1) % (size + 1) == 0){
            continue;
        }
        in >> matrix_ab[i];
    }

    // Reading vector B
    for(auto i = 0u; i < size; ++i){
        in >> matrix_ab[size + i * (size + 1)];
    }

    in.close();

    return matrix_ab;
}

void SaveToFile(double* AB, int size)
{
    std::ofstream out("out.txt", std::ios_base::out);

    out << size << '\n';

    for(auto i = 0u; i < size*(size + 1); ++i){
        if( (i + 1) % (size + 1) == 0){
            out << "\n";
            continue;
        }
        out << AB[i] << '\t';
    }

    for(auto i = 0u; i < size; ++i){
        out << AB[size + i * (size + 1)] << '\t';
    }
    out << "\n";

    out.close();
}

void Print(double* AB, const int& size)
{
    std::cout << size << '\n';

    for(auto i = 0u; i < size*(size + 1); ++i){
        if( (i + 1) % (size + 1) == 0){
            std::cout << "\n";
            continue;
        }
        std::cout << AB[i] << "   ";
    }

    for(auto i = 0u; i < size; ++i){
        std::cout << AB[size + i * (size + 1)] << "   ";
    }
    std::cout << "\n";
}

#include <iostream>

#include "Kernel.cuh"

#include "Utilities.hpp"


int main(int argc, char** argv)
{
    if(argc > 1){
        int size;
        std::cout << argv[1] << '\n';
        double* AB = LoadFromFile(size, argv[1]);

        Print(AB, size);

        GaussEliminationSolver(AB, size, size + 1);

        std::cout << "\n\n";

        Print(AB, size);
    }
    return 0;
}
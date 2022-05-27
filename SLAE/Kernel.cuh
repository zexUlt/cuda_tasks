#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define THREADS_PER_BLOCK 16

void GaussEliminationSolver(double *AB, int rows, int cols);
__global__ void CUDA_ReplaceZero(double* AB, int rows, int columns, int column);
__global__ void CUDA_ColumnElimination(double* AB, int rows, int columns, int column);
__global__ void CUDA_ReverseRowElimination(double* AB, int rows, int columns, int row);
__global__ void CUDA_MultiplyColumn(double* AB, int rows, int columns, int row);

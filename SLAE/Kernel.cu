#include "Kernel.cuh"

void GaussEliminationSolver(double *AB, int rows, int cols)
{
    double* AB_gpu;
    int total_size = sizeof(double) * rows * cols;

    cudaMalloc(&AB_gpu, total_size);
    cudaMemcpy(AB_gpu, (void*)AB, total_size, cudaMemcpyHostToDevice);

    int block_size;

    for(int column = 0; column < cols - 1; ++column){
        block_size = (cols - column - 1)/THREADS_PER_BLOCK + 1;
        
        CUDA_ReplaceZero<<<block_size, THREADS_PER_BLOCK>>>(AB_gpu, rows, cols, column);
        cudaDeviceSynchronize();

        block_size = ((rows - column) * (cols - column) - 1)/THREADS_PER_BLOCK + 1;
        CUDA_ColumnElimination<<<block_size, THREADS_PER_BLOCK>>>(AB_gpu, rows, cols, column);
        cudaDeviceSynchronize();
    }

    for(int row = rows - 1; row >= 0; --row){
        CUDA_ReverseRowElimination<<<1, cols>>>(AB_gpu, rows, cols, row);
        CUDA_MultiplyColumn<<<1, rows>>>(AB_gpu, rows, cols, row);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(AB, (void*)AB_gpu, total_size, cudaMemcpyDeviceToHost);

    cudaFree(AB_gpu);
}

__global__ void CUDA_ReplaceZero(double* AB, int rows, int columns, int column)
{
    if(fabs(AB[column * columns + column]) <= 1e-4){
        int row = column;
        for(; row < rows; ++row){
            if(fabs(AB[row * columns + column]) > 1e-4){
                break;
            }
        }

        int threadId = blockDim.x * blockIdx.x + threadIdx.x;
        if(threadId + column >= columns){
            return;
        }

        int zero = column * columns + column + threadId;
        int chosen = row * columns + column + threadId;
        AB[zero] += AB[chosen]; 
    }
}

__global__ void CUDA_ColumnElimination(double* AB, int rows, int columns, int column)
{
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    if(threadId >= (rows - 1 - column) * (columns - column)){
        return;
    }

    int elementRow = column + threadId / (columns - column) + 1;
    int elementCol = column + threadId % (columns - column);
    int elementIdx = elementCol + elementRow * columns;
    int upperElIdx = elementCol + column * columns;

    int mainDiagElement = column + column * columns;
    int mainDiagElement2 = column + elementRow * columns;

    double ratio = AB[mainDiagElement2] / AB[mainDiagElement];

    AB[elementIdx] -= ratio * AB[upperElIdx];
}

__global__ void CUDA_ReverseRowElimination(double* AB, int rows, int columns, int row)
{
    int threadId = threadIdx.x;
    int cols = columns - 2 - row;

    int start_index = row * columns + row + 1;

    int j = cols % 2;
    for(int i = cols / 2; i > 0; i /= 2){
        if(threadId >= i){
            return;
        }

        AB[start_index + threadId] += AB[start_index + threadId + i + j];
        AB[start_index + threadId + i + j] = 0;
        if(j == 1){
            ++i;
        }
        j = i % 2;
        __syncthreads();
    }

    int xElementIdx = (row + 1) * columns - 1;
    int diagElementIdx = row * columns + row;

    if(diagElementIdx + 1 != xElementIdx){
        AB[xElementIdx] -= AB[diagElementIdx + 1];
        AB[diagElementIdx + 1] = 0.;
    }

    AB[xElementIdx] /= AB[diagElementIdx];
    AB[diagElementIdx] = 1.;
}

__global__ void CUDA_MultiplyColumn(double* AB, int rows, int columns, int row)
{
    AB[(threadIdx.x * columns) + row] *= AB[columns * (row + 1) - 1];
}
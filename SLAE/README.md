# SLAE Solver

The system of linear equations solver

## Building

```
$ nvcc main.cu Utilities.cpp Kernel.cu -o SLAESolver.out
```

## Usage
The program prints the result into stdout
```
./SLAESolver.out input
```

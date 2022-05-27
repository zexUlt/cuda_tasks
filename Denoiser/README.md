# Denoiser

Image denoising using K Nearest Neighbors filter implemented in CUDA technology

## Building
Denoiser uses `libpng` to work with `.png` files 
```
$ nvcc ./image_denoiser.cu -lpng -o denoiser.out
```

## Usage
Denoiser takes two command-line arguments
  + Input file
  + Output file

```
$ ./denoiser.out ./images_raw/portrait_noise.png ./images_fixed/portrait_denoised.png
```

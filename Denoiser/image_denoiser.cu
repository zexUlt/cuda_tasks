#include <cstdlib>
#include <iostream>
#include <cmath>
#include <string>

#include <png.h>

#include <cuda_runtime.h>

namespace constants{
    const int filter_radius     = 3; // M
    const int filter_area       = ((2 * filter_radius + 1) * (2 * filter_radius + 1)); // N^2
    const float inv_filter_area = (1.f / static_cast<float>(filter_area)); // (1 / r^2)
    const float weight_thresh   = .02f;
    const float lerp_thresh     = .66f;
    const float noise_val       = .32f;
    const float noise           = (1.f / (noise_val * noise_val)); // (1 / h^2)
    const float lerpc           = .16f;
}; // end namespace constants

namespace globals{
    int width;
    int height;
    png_byte ctype;
    png_byte bit_depth;
    png_bytep* row_pointers = nullptr;
}; // end namespace globals


__host__ void ReadPngFile(const char* filename)
{
    FILE* fp = fopen(filename, "rb");

    if(!fp){
        abort();
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if(!png){
        abort();
    }    

    png_infop info = png_create_info_struct(png);

    if(!info){
        abort();
    }

    if(setjmp(png_jmpbuf(png))){
        abort();
    }

    png_init_io(png, fp);

    png_read_info(png, info);

    globals::width     = png_get_image_width(png, info);
    globals::height    = png_get_image_height(png, info);
    globals::ctype     = png_get_color_type(png, info);
    globals::bit_depth = png_get_bit_depth(png, info);

    // Read any color_type into 8but depth, RBGA format
    if(globals::bit_depth == 16){
        png_set_strip_16(png);
    }

    if(globals::ctype == PNG_COLOR_TYPE_PALETTE){
        png_set_palette_to_rgb(png);
    }

    if(globals::ctype == PNG_COLOR_TYPE_GRAY && globals::bit_depth < 8){
        png_set_expand_gray_1_2_4_to_8(png);
    }

    if(png_get_valid(png, info, PNG_INFO_tRNS)){
        png_set_tRNS_to_alpha(png);
    }

    if(globals::ctype == PNG_COLOR_TYPE_RGB  ||
       globals::ctype == PNG_COLOR_TYPE_GRAY || 
       globals::ctype == PNG_COLOR_TYPE_PALETTE)
    {
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    }

    if(globals::ctype == PNG_COLOR_TYPE_GRAY || 
       globals::ctype == PNG_COLOR_TYPE_GRAY_ALPHA)
    {
        png_set_gray_to_rgb(png);
    }

    png_read_update_info(png, info);

    if(globals::row_pointers){
        abort();
    }

    globals::row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * globals::height);

    for(int y = 0; y < globals::height; ++y){
        globals::row_pointers[y] = (png_bytep)malloc(png_get_rowbytes(png, info));
    }

    png_read_image(png, globals::row_pointers);

    fclose(fp);
    
    png_destroy_read_struct(&png, &info, NULL);
}

__host__ void WritePngFile(const char* filename)
{
    FILE* fp = fopen(filename, "wb");

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if(!png){
        abort();
    }

    png_infop info = png_create_info_struct(png);
    if(!info){
        abort();
    }

    if(setjmp(png_jmpbuf(png))){
        abort();
    }

    png_init_io(png, fp);

    // Output is 8bit depth, RGBA format
    png_set_IHDR(
        png, 
        info,
        globals::width, globals::height,
        8,
        PNG_COLOR_TYPE_RGBA,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );

    png_write_info(png, info);

    if(!globals::row_pointers){
        abort();
    }

    png_write_image(png, globals::row_pointers);
    png_write_end(png, NULL);

    for(int y = 0; y < globals::height; ++y){
        free(globals::row_pointers[y]);
    }

    free(globals::row_pointers);

    fclose(fp);

    png_destroy_write_struct(&png, &info);
}


__host__ void ImageToArray(png_bytep image)
{
    for(int y = 0; y < globals::height; ++y){
        png_bytep row = globals::row_pointers[y];
        for(int x = 0; x < globals::width; ++x){
            png_bytep px = &(row[x * 4]);
            image[(y * globals::width + x) * 4] = px[0];
            image[(y * globals::width + x) * 4 + 1] = px[1];
            image[(y * globals::width + x) * 4 + 2] = px[2];
            image[(y * globals::width + x) * 4 + 3] = px[3];
        }
    }
}

__host__ void ArrayToImage(png_bytep image)
{
    for(int y = 0; y < globals::height; ++y){
        png_bytep row = globals::row_pointers[y];
        for(int x = 0; x < globals::width; ++x){
            png_bytep px = &(row[x * 4]);
            px[0] = image[(y * globals::width + x) * 4];
            px[1] = image[(y * globals::width + x) * 4 + 1];
            px[2] = image[(y * globals::width + x) * 4 + 2];
            px[3] = image[(y * globals::width + x) * 4 + 3];
        }
    }
}

__host__ void ExitWithMessage(const std::string&& msg)
{
    std::cerr << msg << '\n';
    std::exit(EXIT_FAILURE);
}

__device__ float CUDA_PixelDistanceSquared(float x, float y)
{
    return x * x + y * y;
}

__device__ float CUDA_ColorDistanceSquared(float4 c1, float4 c2)
{
    return (
        (c2.x - c1.x) / 255.f * (c2.x - c1.x) / 255.f + 
        (c2.y - c1.y) / 255.f * (c2.y - c1.y) / 255.f + 
        (c2.z - c1.z) / 255.f * (c2.z - c1.z) / 255.f
    ); 
}

__device__ float Lerpf(float start, float end, float step)
{
    return start + (end - start) * step;
}

__global__ void knn_filter(png_bytep img, png_bytep img_out, int width, int height)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx >= width || idy >= height){
        return;
    }

    // Normalized counter for the weight threshold
    float fCount = 0;

    // Total sum of pixel weights
    float weights_sum = 0;

    // Result accumulator
    float3 color{0, 0, 0};

    // Center of the filter
    int pos = (idy * width + idx) * 4;
    float4 color_center{
        float(img[pos]), float(img[pos + 1]), 
        float(img[pos + 2]), float(img[pos + 3])
    };

    for(int y = -constants::filter_radius; y <= constants::filter_radius; ++y){
        for(int x = -constants::filter_radius; x <= constants::filter_radius; ++x){
            if(idy + y < 0 || idy + y >= height || idx + x < 0 || idx + x >= width){
                continue;
            }

            int curr_pos = ((idy + y) * width + (idx + x)) * 4;
            float4 color_xy{
                float(img[curr_pos]), float(img[curr_pos + 1]),
                float(img[curr_pos + 2]), float(img[curr_pos + 3])
            };

            float pixel_distance = CUDA_PixelDistanceSquared(float(x), float(y));

            float color_distance = CUDA_ColorDistanceSquared(color_center, color_xy);

            // Denoising
            float weight_xy = expf(
                -(pixel_distance * constants::inv_filter_area + 
                color_distance * constants::noise)
            );

            color.x += color_xy.x * weight_xy;
            color.y += color_xy.y * weight_xy;
            color.z += color_xy.z * weight_xy;

            weights_sum += weight_xy;
            fCount += (weight_xy > constants::weight_thresh) ? constants::inv_filter_area : 0;
        }
    }

    // Normalize result color
    weights_sum = 1.f / weights_sum;
    color.x *= weights_sum;
    color.y *= weights_sum;
    color.z *= weights_sum;

    float lerpQ = (fCount > constants::lerp_thresh) ? constants::lerpc : 1.f - constants::lerpc;

    color.x = Lerpf(color.x, color_center.x, lerpQ);
    color.y = Lerpf(color.y, color_center.y, lerpQ);
    color.z = Lerpf(color.z, color_center.z, lerpQ);

    // Result to memory
    img_out[pos]     = png_byte(color.x);
    img_out[pos + 1] = png_byte(color.y);
    img_out[pos + 2] = png_byte(color.z);
    img_out[pos + 3] = img[pos + 3];
}


int main(int argc, char** argv)
{
    png_bytep host_img;

    // Read png file to an array
    ReadPngFile(argv[1]); 

    size_t size = globals::width * globals::height * sizeof(png_byte) * 4;

    cudaMallocHost((void**)(&host_img), size);

    png_byte* device_img = nullptr;
    if(cudaSuccess != cudaMalloc((void**)(&device_img), size)){
        ExitWithMessage("Error allocating memory on the GPU 1");
    }

    png_byte* device_output = nullptr;
    if(cudaSuccess != cudaMalloc((void**)(&device_output), size)){
        ExitWithMessage("Error allocating memory on the GPU 2");
    }

    // Copy image to allocated array
    ImageToArray(host_img);

    // Copy image array to device
    if(cudaSuccess != cudaMemcpy(device_img, host_img, size, cudaMemcpyHostToDevice)){
        ExitWithMessage("Error copying data to device");
    }

    // Kernel block/thread config
    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks(ceil(static_cast<float>(globals::width) / threadsPerBlock.x),
                   ceil(static_cast<float>(globals::height) / threadsPerBlock.y));

    // Kernel
    knn_filter<<<numBlocks, threadsPerBlock>>>(device_img, device_output, globals::width, globals::height);

    if(cudaSuccess != cudaGetLastError()){
        ExitWithMessage("Error during kernel launch");
    }

    cudaDeviceSynchronize();

    // Copy memory back to host
    if(cudaSuccess != cudaMemcpy(host_img, device_output, size, cudaMemcpyDeviceToHost)){
        ExitWithMessage("Error copying results to host");
    }

    // Prepare array to write png
    ArrayToImage(host_img);

    // Release resorces
    if(cudaSuccess != cudaFreeHost(host_img)){
        ExitWithMessage("Error when deallocating space on host");
    }
    if(cudaSuccess != cudaFree(device_img)){
        ExitWithMessage("Error when deallocating space on the GPU");
    }
    if(cudaSuccess != cudaFree(device_output)){
        ExitWithMessage("Error when deallocating output space on the GPU");
    }

    // Write array to new png file
    WritePngFile(argv[2]);

    return 0;
}
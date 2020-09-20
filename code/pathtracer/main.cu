#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include "include/image.cuh"
#include "cuda_profiler_api.h"
#include "include/sceneparser.cuh"
#include "include/pathtracer.cuh"


__global__ void renderPixel(int samps, Camera* camera, Objects* scene, Vector3f* device_output) {

    uint h = camera->getHeight();
    uint w = camera->getWidth();

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    assert(x < w && y < h);
    //printf("Thread Start at (%d,%d)\n",x,y);
    //init resource
    ThreadResource local_resource = scene->getResource();
    local_resource._repeat_mark = (BoolBitField*)malloc(sizeof(BoolBitField) * local_resource._mark_size);

    if(!local_resource._repeat_mark) {
        printf("Malloc failed, quit.\n");
        return;
    }
    uint seeds[2] = {x, y};
    Vector3f pixel_color = make_float3(0,0,0);
    for (uint sy = 0; sy < 2; sy++) {
        for (uint sx = 0; sx < 2; sx++) {
            Vector3f subpixel_color = make_float3(0,0,0);
            for (uint s = 0; s < samps; s++) {
                const Ray camRay = camera->getRay(x, y, sx, sy, seeds);
                //const Ray camRay = camera->getRay(516, 163, sx, sy, seeds);
                subpixel_color += radiance<0>(camRay, scene, seeds, &local_resource) * (1. / samps);
            }
            pixel_color += make_float3(Image::clamp(subpixel_color.x), Image::clamp(subpixel_color.y), Image::clamp(subpixel_color.z)) * .25f;
        }
    }

    free(local_resource._repeat_mark);

    uint i = (h - y - 1) * w + x;
    device_output[i] = pixel_color;
}

//TODO: set memories on Shared or constant mem
int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <ssp> <scene_file>", argv[0]);
        exit(0);
    }

    int samps = atoi(argv[1]) / 4;
    assert(samps > 0);

    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize,1024*1024*1024));
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize,128*1024));
    checkCudaErrors(cudaProfilerStart());


    timestamp(stamp0);
    /* Create scene */
    auto *sceneParser = new SceneParser(argv[2]);
    auto camera = sceneParser->getCamera();
    auto w = camera->getWidth();
    auto h = camera->getHeight();
    auto objs = sceneParser->getGroup();
    checkCudaErrors(cudaDeviceSynchronize());


    timestamp(stamp1);
    auto *host_output = new Vector3f[w * h];
    Vector3f *device_output;
    cudaMalloc(&device_output, w * h * sizeof(Vector3f));
    checkCudaErrors(cudaDeviceSynchronize());


    /* Launch kernel */
    timestamp(stamp2);
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, renderPixel, 0, w * h));
    dim3 block(sqrt(blockSize), sqrt(blockSize), 1);
    dim3 grid(w / block.x, h / block.y, 1);

    printf("CUDA initialized. \nStart rendering...\n");
    checkCudaErrors(cudaDeviceSynchronize());


    timestamp(stamp3)
    renderPixel<<<grid, block>>>(samps, camera, objs, device_output);
    //renderPixel<<<1, 1>>>(samps, camera, objs, device_output);

    checkCudaErrors(cudaDeviceSynchronize());
    timestamp(stamp4);


    checkCudaErrors(cudaMemcpy(host_output, device_output, w*h*sizeof(Vector3f),cudaMemcpyDeviceToHost));
    Image::savePPM("", host_output, w, h);
    checkCudaErrors(cudaDeviceSynchronize());

    timestamp(stamp5);











    std::cout << "Create material and objects elapsed: " << getDuration(stamp0, stamp1) << std::endl;
    std::cout << "Create output mat elapsed:           " << getDuration(stamp1, stamp2) << std::endl;
    std::cout << "Initial CUDA block and grid elapsed: " << getDuration(stamp2, stamp3) << std::endl;
    std::cout << "Render elapsed:                      " << getDuration(stamp3, stamp4) << std::endl;
    std::cout << "Copy from device to host elapsed:    " << getDuration(stamp4, stamp5) << std::endl;



    cudaFree(device_output);
    delete[] host_output;
    delete sceneParser;

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaProfilerStop());
    checkCudaErrors(cudaDeviceReset());
    return 0;
}

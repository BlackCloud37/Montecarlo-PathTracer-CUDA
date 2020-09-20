//
// Created by blackcloud on 2020/5/5.
//

#ifndef CUDA_TEST_UTILS_CUH
#define CUDA_TEST_UTILS_CUH
#include "objects.cuh"
#define timestamp(__var__) auto __var__ = std::chrono::system_clock::now();

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
	_where << __FILE__ << ':' << __LINE__;                             \
	_message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
	std::cerr << _message.str() << "\nAborting...\n";                  \
	cudaDeviceReset();                                                 \
	exit(1);                                                           \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define printVector(vec) do {                                          \
    printf("%f %f %f", vec.x, vec.y, vec.z);                           \
} while(0)

inline double getDuration(std::chrono::time_point<std::chrono::system_clock> a,
                          std::chrono::time_point<std::chrono::system_clock> b) {
    return  std::chrono::duration<double>(b - a).count();
}
__host__ __device__ inline static float rand(uint *seed) {
    seed[0] = 36969 * (seed[0] & 65535) + (seed[0] >> 16);
    seed[1] = 18000 * (seed[1] & 65535) + (seed[1] >> 16);

    uint ires = (seed[0] << 16) + (seed[1]);

    union
    {
        float f;
        uint ui;
    } res;

    res.ui = (ires & 0x007fffff) | 0x40000000;  // bitwise AND, bitwise OR

    return (res.f - 2.f) / 2.f;
}


__device__ float3 mult(const float3& f1, const float3& f2) {
    return make_float3(f1.x*f2.x, f1.y*f2.y, f1.z*f2.z);
}



#endif //CUDA_TEST_UTILS_CUH

//
// Created by blackcloud on 2020/5/5.
//

#ifndef CUDA_TEST_MANAGED_CUH
#define CUDA_TEST_MANAGED_CUH
class Managed {
public:
    __host__ void *operator new(size_t len) {
        //printf("Managed\n");
        void *ptr;
        cudaMallocManaged(&ptr, len);
        auto err = cudaGetLastError();
        if (err != 0) {printf("ErrorHere %s\n", cudaGetErrorName(err));}
        return ptr;
    }

    void operator delete(void *ptr) {
        cudaFree(ptr);
    }
};
#endif //CUDA_TEST_MANAGED_CUH

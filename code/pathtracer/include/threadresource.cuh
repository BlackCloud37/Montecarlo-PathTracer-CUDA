//
// Created by blackcloud on 2020/5/8.
//

#ifndef CUDA_TEST_THREADRESOURCE_CUH
#define CUDA_TEST_THREADRESOURCE_CUH

struct BoolBitField {
    bool b0:1;
    bool b1:1;
    bool b2:1;
    bool b3:1;
    bool b4:1;
    bool b5:1;
    bool b6:1;
    bool b7:1;
    __device__ bool operator[](int index) {
        switch (index) {
            case 0:
                return b0;
            case 1:
                return b1;
            case 2:
                return b2;
            case 3:
                return b3;
            case 4:
                return b4;
            case 5:
                return b5;
            case 6:
                return b6;
            case 7:
                return b7;
            default:
                return false;
        }
    }
};

struct ThreadResource {
    // For mesh
    int _mark_size = 0;
    BoolBitField *_repeat_mark;

    // For curve
    int _base_size = 0;
    float *_base;
    int _derivative_size = 0;
    float *_derivative;
    int _s_size = 0;
    float *_s;
    int _ds_size = 0;
    float *_ds;

    __device__ __host__ ThreadResource() {
        _mark_size = 0;
        _repeat_mark = nullptr;

        _base_size = 0;
        _base = nullptr;
        _derivative_size = 0;
        _derivative = nullptr;
        _s_size = 0;
        _s = nullptr;
        _ds_size = 0;
        _ds = nullptr;
    }
    __device__ __host__ ThreadResource(const ThreadResource& other) {
        _mark_size = other._mark_size;
        _base_size = other._base_size;
        _derivative_size = other._derivative_size;
        _s_size = other._s_size;
        _ds_size = other._ds_size;
    }
};
#endif //CUDA_TEST_THREADRESOURCE_CUH

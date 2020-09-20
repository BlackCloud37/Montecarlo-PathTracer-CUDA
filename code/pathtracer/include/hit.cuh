//
// Created by blackcloud on 2020/5/5.
//

#ifndef CUDA_TEST_HIT_CUH
#define CUDA_TEST_HIT_CUH
#include "vec/vec_mat.cuh"
#include "material.cuh"
class Material;
struct MaterialFeature;



class Hit {
public:

    // constructors
    __host__ __device__ Hit() {
        //material = nullptr;
        t = 1e38;
    }

    __device__ Hit(float _t, const MaterialFeature& mf, const Vector3f& n) {
        t = _t;
        feature = mf;
        normal = n;
    }
    __host__ __device__ Hit(const Hit &h) {
        t = h.t;
        feature = h.feature;
        normal = h.normal;
    }

    // destructor
    ~Hit() = default;

    __device__ double getT() const {
        return t;
    }
    __device__ const MaterialFeature& getFeature() const {
        return feature;
    }

    __device__ const Vector3f &getNormal() const {
        return normal;
    }
    __device__ void set(double _t, const MaterialFeature& mf, const Vector3f &n) {
        t = _t;
        feature = mf;
        normal = n;
    }


private:
    double t;
    Vector3f normal;
    MaterialFeature feature;

//    int id;
//    HitType hit_type;
//    Vector3f beta;
//    union {
//
//    };

};

//__device__ inline std::ostream &operator<<(std::ostream &os, const Hit &h) {
//    os << "Hit <" << h.getT() << ", " << h.getNormal() << ">";
//    return os;
//}
#endif //CUDA_TEST_HIT_CUH

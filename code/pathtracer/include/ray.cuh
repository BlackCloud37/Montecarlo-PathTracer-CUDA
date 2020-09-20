//
// Created by blackcloud on 2020/5/5.
//

#ifndef CUDA_TEST_RAY_CUH
#define CUDA_TEST_RAY_CUH
#include <cassert>
#include <iostream>
#include "vec/vec_mat.cuh"
#include "managed.cuh"

// Ray class mostly copied from Peter Shirley and Keith Morley
class Ray : public Managed {
public:

    Ray() = delete;
    __host__ __device__ Ray(const Ray& r) {
        origin = r.origin;
        direction = r.direction;
    }
    __host__ __device__ Ray(const Vector3f &orig, const Vector3f &dir) {
        origin = orig;
        direction = normalize(dir);
    }

//    __host__ __device__ Ray(const Ray &r) {
//        origin = r.origin;
//        direction = r.direction;
//    }

//    __host__ __device__ Ray(float3 orig, float3 dir) {
//        origin = orig;
//        direction = normalize(dir);
//    }

    __host__ __device__ const Vector3f &getOrigin() const {
        return origin;
    }

    __host__ __device__ const Vector3f &getDirection() const {
        return direction;
    }

    __host__ __device__ Vector3f pointAtParameter(float t) const {
        return origin + direction * t;
    }

private:
    Vector3f origin;
    Vector3f direction;

};

//inline std::ostream &operator<<(std::ostream &os, const Ray &r) {
//    os << "Ray <" << r.getOrigin() << ", " << r.getDirection() << ">";
//    return os;
//}
#endif //CUDA_TEST_RAY_CUH

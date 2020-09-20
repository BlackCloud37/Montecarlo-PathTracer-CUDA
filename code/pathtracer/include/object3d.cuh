//
// Created by blackcloud on 2020/5/5.
//

#ifndef CUDA_TEST_OBJECT3D_CUH
#define CUDA_TEST_OBJECT3D_CUH
#include "ray.cuh"
#include "hit.cuh"
#include "material.cuh"
#include "aabb.cuh"
#include "threadresource.cuh"

// Base class for all 3d entities.
class Object3D {
public:
    Object3D() = delete ;

    __device__ Object3D(const Object3D& o) {
        material = o.material;
        if (!o.box) {
            this->box = new AABB(o.box->getMin(), o.box->getMax());
        }
    }

    __device__ explicit Object3D(Material *material) {
        this->material = material;
        this->box = nullptr;
    }
    __device__ virtual ~Object3D() {
        delete box;
    }
    // Intersect Ray with this object. If hit, store information in hit structure.
    __device__ virtual bool intersect(const Ray &r, Hit &h, float tmin, ThreadResource* thread_resource) const = 0;
    __host__ __device__ AABB* getBox() const {
        return box;
    }
protected:
    AABB *box;
    Material *material;
};

#endif //CUDA_TEST_OBJECT3D_CUH

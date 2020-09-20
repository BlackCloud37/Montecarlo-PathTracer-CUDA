//
// Created by blackcloud on 2020/5/7.
//

#ifndef CUDA_TEST_MESH_CUH
#define CUDA_TEST_MESH_CUH

#include "kdtree.cuh"
//#include ""


class Mesh : public Object3D {
    KdTree* kdtree;
public:
    __device__ Mesh(KdTree* _kdtree, Material* m) : Object3D(m), kdtree(_kdtree) {
        box = new AABB(kdtree->getRoot()->getBox());
    }
    __device__ ~Mesh() override {
        //delete box;
    }
    __device__ bool intersect(const Ray &r, Hit &h, float tmin, ThreadResource* thread_resource) const override {
        //printf("Calling mesh.intersect\n");
        //printf("MeshInter\n");
        bool result = kdtree->intersect(r, h, tmin, thread_resource);
        if (result) {
            h.set(h.getT() , material->getMaterialFeatureAtPoint(0,0), h.getNormal());
        }
        return result;
    }
};
#endif //CUDA_TEST_MESH_CUH

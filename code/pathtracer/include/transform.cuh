//
// Created by blackcloud on 2020/5/7.
//

#ifndef CUDA_TEST_TRANSFORM_CUH
#define CUDA_TEST_TRANSFORM_CUH

#include "object3d.cuh"
#define printVector(vec) do {                                          \
    printf("%f %f %f", vec.x, vec.y, vec.z);                           \
} while(0)

// TODO: implement this class so that the intersect function first transforms the ray
class Transform : public Object3D {
public:
    Transform() = delete ;

    __device__ Transform(const Matrix4f& m, Object3D *obj) : o(obj), Object3D(nullptr) {
        transform = m.inverse();
        auto tmi = transformPoint(m, obj->getBox()->getMin());
        auto tma = transformPoint(m, obj->getBox()->getMax());

        box = new AABB(tmi, tma);// TODO:Problem here?
        printf("Transfrom:");
        for (int i = 0; i < 3; i ++) {
            for (int j = 0; j < 3; j++) {
                printf("%f ", transform(i,j));
            }
            printf("\n");
        }
        printf("\n");
//        printVector(tmi);
//        printf("\n");
//        printVector(tma);
//        printf("\n");
        //box = new AABB(make_float3(-3000,-3000,-3000), make_float3(300,300,300));
    }

    __device__ ~Transform() override {
        //delete box;
        delete o;
    }


    __device__ bool intersect(const Ray &r, Hit &h, float tmin, ThreadResource* thread_resource) const override {
        //printf("TransInter\n");

        float3 trSource = transformPoint(transform, r.getOrigin());
        float3 trDirection = transformDirection(transform, r.getDirection());
        Ray tr(trSource, trDirection);
        bool inter = o->intersect(tr, h, tmin, thread_resource);
        if (inter) {
            Vector3f normal = normalize(transformDirection(transform.transposed(), h.getNormal()));
            //printf("Transhit at Point ");
            //Vector3f p = r.pointAtParameter(h.getT() * (length(r.getDirection())/length(trDirection)));
            //printf("%f %f %f\n\n\n", p.x, p.y, p.z);
            //printf("Emission %f %f %f\n\n", h.getFeature().getEmission().x, h.getFeature().getEmission().y, h.getFeature().getEmission().z);
            h.set(h.getT() * (length(r.getDirection())/length(trDirection)), h.getFeature(), normal);
        }
        return inter;
    }

protected:
    Object3D *o; //un-transformed object
    Matrix4f transform;
};
#endif //CUDA_TEST_TRANSFORM_CUH

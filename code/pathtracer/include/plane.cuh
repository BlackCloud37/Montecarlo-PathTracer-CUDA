//
// Created by blackcloud on 2020/5/6.
//

#ifndef CUDA_TEST_PLANE_CUH
#define CUDA_TEST_PLANE_CUH
//#include "material.cuh"
#include "object3d.cuh"
#include "utils.cuh"
// Implement Plane representing an infinite plane
// function: ax+by+cz=d
// choose your representation , add more fields and fill in the functions

class Plane : public Object3D {
public:
    Plane() = delete ;
    __device__ Plane(const Plane& p) : Object3D(p.material){
        d = p.d;
        normal = p.normal;
    }
    __device__ Plane(const Vector3f &normal, float d, const int range, const Vector3f& tex_centerp, const Vector3f& texDirU, const Vector3f& texDirV, Material *m) :Object3D(m) {
        this->d = d;
        this->normal = normalize(normal);
        Vector3f centerp = normal * d/dot(normal, normal);
        Vector3f minp = centerp - make_float3(abs(range), abs(range), abs(range)) + make_float3(abs(normal.x) ,abs(normal.y), abs(normal.z)) * (abs(range)-.5);
        Vector3f maxp = centerp + make_float3(abs(range), abs(range), abs(range)) - make_float3(abs(normal.x) ,abs(normal.y), abs(normal.z)) * (abs(range)-.5);
        //TODO:
        texOri = tex_centerp;
        this->texDirV = texDirV;
        this->texDirU = texDirU;
        this->box = new AABB(minp, maxp);
    }
    __device__ Plane(const Vector3f &normal, float d, const Vector3f &minp, const Vector3f& maxp, Material *m) :Object3D(m) {
        this->d = d;
        this->normal = normalize(normal);
        this->box = new AABB(minp, maxp);
    }

    __device__ bool intersect(const Ray &ray, Hit &hit, float tmin, ThreadResource* thread_resource) const override {
        //printf("PlaneInter\n");
        double dn = dot(normal, ray.getDirection());
        if (dn > 0) return false;
        auto t = (d-dot(normal, ray.getOrigin())) / dn;
        if (t > hit.getT() || t < tmin + 0.0125) return false;
        if (material->ifHasTexture()) {
            float u, v;
            auto p = ray.pointAtParameter(t);
            Vector3f vec = p - texOri;
            u = dot(vec, texDirU);
            u = abs(floor(u) - u);
            v = dot(vec, texDirV);
            v = abs(floor(v) - v);

            MaterialFeature mf = material->getMaterialFeatureAtPoint(u,v);
            auto n1 = normalize((1-abs(mf.gradientU)) * normal + normalize(texDirU) * mf.gradientU);
            auto ng = normalize((1-abs(mf.gradientV)) * n1 + normalize(texDirV) * mf.gradientV);
            hit.set(t, mf, ng);
        } else {
            hit.set(t, material->getMaterialFeatureAtPoint(0,0), normal);
        }
        return true;
    }

public:
    float d;
    Vector3f normal;
    Vector3f texOri, texDirU, texDirV;
    //float texScaleU, texScaleV;
};
#endif //CUDA_TEST_PLANE_CUH

//
// Created by blackcloud on 2020/5/5.
//

#ifndef CUDA_TEST_SPHERE_CUH
#define CUDA_TEST_SPHERE_CUH
#include "object3d.cuh"

//Implement functions and add more fields as necessary

class Sphere : public Object3D {
public:

    Sphere() = delete;
    __device__ Sphere(const Sphere& s) : Object3D(s.material) {
        center = s.center;
        radius = s.radius;
        box = new AABB(s.box->getMin(), s.box->getMax());
    }

    __device__ Sphere(const Vector3f &center, float radius, Material *material) : Object3D(material) {
        this->center = center;
        this->radius = radius;
        const Vector3f& radbox = make_float3(radius,radius,radius);
        box = new AABB(center - radbox, center + radbox);
    }

    __device__ bool intersect(const Ray &ray, Hit &hit, float tmin, ThreadResource* thread_resource) const override {

        //printf("Intersection test Sphere\n");
        //if (!box->isIntersect(ray)) return false;
        Vector3f op = center-ray.getOrigin();
        float t, eps=0.0125,
        b = dot(op, ray.getDirection()),
        det = b*b - dot(op,op) + radius*radius;

        if (det < 0) // no result
            return false;
        else
            det = sqrt(det);

        t = (t = b - det) > tmin + eps ? t : ((t = b + det) > tmin + eps ? t : tmin);

        if (t <= tmin || t > hit.getT())
            return false;

        const Vector3f& norm = normalize((ray.pointAtParameter(t) - center));

        if (material->ifHasTexture()){
            float u, v;
            u = atan2(norm.x, norm.z) / (2 * M_PI) + .5f;
            v = norm.y * .5f + .5f;
            auto mf = material->getMaterialFeatureAtPoint(u,v);
            auto a = make_float3(0,1,0);
            auto texDirU = a - dot(norm,a)*norm;
            auto texDirV = cross(a,norm);
            auto n1 = normalize((1-abs(mf.gradientU)) * norm + normalize(texDirU) * mf.gradientU);
            auto ng = normalize((1-abs(mf.gradientV)) * n1 + normalize(texDirV) * mf.gradientV);
            hit.set(t, mf, ng);
        } else {
            hit.set(t, material->getMaterialFeatureAtPoint(0,0), norm);
        }

        return true;
    }


public:
    Vector3f center;
    float radius;
    float texScale;
protected:

};
#endif //CUDA_TEST_SPHERE_CUH

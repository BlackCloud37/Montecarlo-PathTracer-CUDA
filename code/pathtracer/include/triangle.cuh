//
// Created by blackcloud on 2020/5/7.
//

#ifndef CUDA_TEST_TRIANGLE_CUH
#define CUDA_TEST_TRIANGLE_CUH

#define VERT_A vertices[0]
#define VERT_B vertices[1]
#define VERT_C vertices[2]
// implement this class and add more fields as necessary,
#include "object3d.cuh"
class Triangle {
public:
    Triangle() = delete;

    // a b c are three vertex positions of the triangle
    __device__ __host__ Triangle( const Vector3f& a, const Vector3f& b, const Vector3f& c) {
        VERT_A = a;
        VERT_B = b;
        VERT_C = c;
        normal = normalize(cross(b-a, c-a));
        d = dot(a, normal);
        centroid = make_float3((VERT_A.x + VERT_B.x + VERT_C.x) / 3.,
                               (VERT_A.y + VERT_B.y + VERT_C.y) / 3.,
                               (VERT_A.z + VERT_B.z + VERT_C.z) / 3.);
    }

    const Vector3f& getCentroid() const {
        return centroid;
    }

    __device__ bool intersect(const Ray& ray,  Hit& hit , float tmin) const {



        double dn = dot(normal, ray.getDirection());
        if (dn > 0) return false;
        auto t = (d-dot(normal, ray.getOrigin())) / dn;
        if (t > hit.getT() || t <= tmin + 0.0125) return false;
        Vector3f vec_P = ray.pointAtParameter(t);
        if (dot(cross(VERT_A - vec_P, VERT_B - vec_P), normal) < 0 ||
            dot(cross(VERT_B - vec_P, VERT_C - vec_P), normal) < 0 ||
            dot(cross(VERT_C - vec_P, VERT_A - vec_P), normal) < 0)
            return false;
        hit.set(t, MaterialFeature(), normal);

//        //debug
//        printf("Hit In tri\n");
//        printf("Ray: %f, %f, %f -> %f, %f, %f\n", ray.getOrigin().x, ray.getOrigin().y, ray.getOrigin().z, ray.getDirection().x, ray.getDirection().y, ray.getDirection().z);
//        printf("Hit Norm: %f, %f, %f\n", normal.x, normal.y, normal.z);
//        printf("Hit t: %f\n\n", t);
//        //

        return true;
    }


public:
    //TODO:remove some var
    float d; // parameter d as the plane has
    Vector3f normal;
    Vector3f centroid;
    Vector3f vertices[3];
protected:

};
#endif //CUDA_TEST_TRIANGLE_CUH

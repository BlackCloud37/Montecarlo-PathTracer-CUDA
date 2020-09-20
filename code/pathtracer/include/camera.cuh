//
// Created by blackcloud on 2020/5/5.
//

#ifndef CUDA_TEST_CAMERA_CUH
#define CUDA_TEST_CAMERA_CUH
#include "ray.cuh"
#include "utils.cuh"
#include <cfloat>
#include <cmath>

class Camera : public Ray {
    int w, h;
    double angle;

    double flength;
    double aperture;

    Vector3f r1, r2;
    Vector3f cx, cy;
public:
    Camera(const Vector3f& _p, const Vector3f& _d, const int _w, const int _h, const double _a, const double _fl, const double _ap) :
            Ray(_p, _d), w(_w), h(_h), angle(_a), flength(_fl), aperture(_ap) {
        uint Xi[2] = {0,1};
        double randx = rand(Xi) - 0.5;
        double randy = rand(Xi) - 0.5;
        double randz = (0.0 - getDirection().x * randx - getDirection().y * randx) / getDirection().z;
        r1 = normalize(make_float3(randx, randy, randz));
        r2 = normalize(cross(r1, getDirection()));

        cx = make_float3(w * angle / h, 0, 0);
        cy = normalize(cross(cx, getDirection())) * angle;
    }
    Camera(const Camera& c) : Ray(c.getOrigin(), c.getDirection()) {
        w = c.w;
        h = c.h;
        angle = c.angle;
        aperture = c.aperture;
        flength = c.flength;
        r1 = c.r1;
        r2 = c.r2;
        cx = c.cx;
        cy = c.cy;

    }
    __host__ void setFOV(const double& _f, const double& _a) {
        flength = _f;
        aperture = _a;
    }
    __device__ __host__ int getWidth() const {
        return w;
    }
    __device__ __host__ int getHeight() const {
        return h;
    }
    __device__ __host__ double getFlength() const {
        return flength;
    }
    __device__ __host__ Vector3f getCx() const {
        return cx;
    }
    __device__ __host__ Vector3f getCy() const {
        return cy;
    }
    __device__ __host__ Vector3f getR1() const {
        return r1;
    }
    __device__ __host__ Vector3f getR2() const {
        return r2;
    }
    __device__ __host__ double getAperture() const {
        return aperture;
    }
    __device__ Ray getRay(const int x, const int y, const int sx, const int sy, uint* Xi) const {
        double rp1 = 2 * rand(Xi),
                dx = rp1 < 1 ? sqrt(rp1) - 1 : 1 - sqrt(2 - rp1);
        double rp2 = 2 * rand(Xi),
                dy = rp2 < 1 ? sqrt(rp2) - 1 : 1 - sqrt(2 - rp2);
        Vector3f d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) +
                     cy * (((sy + .5 + dy) / 2 + y) / h - .5) +
                     getDirection();

        if (flength == 0) {
            return Ray(getOrigin(), d);
        }
        else {
            d = normalize(d) * flength;
            double rand1 = rand(Xi) * 2 - 1.;
            double rand2 = rand(Xi) * 2 - 1.;

            Vector3f v1 = r1 * rand1 * aperture;
            Vector3f v2 = r2 * rand2 * aperture;

            Vector3f sp = getOrigin() + v1 + v2;
            Vector3f fp = d + getOrigin();
            d = fp - sp;
            return Ray(sp, d);
        }
    }
    ~Camera() = default;
};
#endif //CUDA_TEST_CAMERA_CUH

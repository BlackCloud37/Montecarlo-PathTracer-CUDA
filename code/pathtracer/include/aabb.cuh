//
// Created by blackcloud on 2020/5/7.
//

#ifndef CUDA_TEST_AABB_CUH
#define CUDA_TEST_AABB_CUH
#include "vec/vec_mat.cuh"
#include "ray.cuh"
#include "triangle.cuh"

class AABB {
    Vector3f min_point, max_point, center, boxHalfSize;
public:
    AABB() {};
    __host__ __device__ AABB(const Vector3f& _min, const Vector3f& _max) : min_point(_min), max_point(_max) {
        boxHalfSize = (max_point - min_point) / 2;
        center = min_point + boxHalfSize;
    }
    __host__ __device__ AABB(const AABB& a) : min_point(a.min_point), max_point(a.max_point), boxHalfSize(a.boxHalfSize), center(a.center){}
    __host__ __device__ bool isIntersect(const Ray& ray) const {
        const Vector3f& origin = ray.getOrigin();
        const Vector3f& dir = ray.getDirection();
        const Vector3f& dirFrac = make_float3(1 / dir.x, 1 / dir.y, 1 / dir.z);

        float t1 = (min_point.x - origin.x) * dirFrac.x;
        float t2 = (max_point.x - origin.x) * dirFrac.x;

        float t3 = (min_point.y - origin.y) * dirFrac.y;
        float t4 = (max_point.y - origin.y) * dirFrac.y;

        float t5 = (min_point.z - origin.z) * dirFrac.z;
        float t6 = (max_point.z - origin.z) * dirFrac.z;

        float tmin = fmax(fmax(fmin(t1, t2), fmin(t3, t4)), fmin(t5, t6));
        float tmax = fmin(fmin(fmax(t1, t2), fmax(t3, t4)), fmax(t5, t6));

        if(tmax < 0) {return false;}
        return tmin <= tmax;
    }
    __device__ bool isIntersect(const Ray& ray, float& t) const {
        const Vector3f& origin = ray.getOrigin();
        const Vector3f& dir = ray.getDirection();
        const Vector3f& dirFrac = make_float3(1 / dir.x, 1 / dir.y, 1 / dir.z);

        float t1 = (min_point.x - origin.x) * dirFrac.x;
        float t2 = (max_point.x - origin.x) * dirFrac.x;

        float t3 = (min_point.y - origin.y) * dirFrac.y;
        float t4 = (max_point.y - origin.y) * dirFrac.y;

        float t5 = (min_point.z - origin.z) * dirFrac.z;
        float t6 = (max_point.z - origin.z) * dirFrac.z;

        float tmin = fmax(fmax(fmin(t1, t2), fmin(t3, t4)), fmin(t5, t6));
        float tmax = fmin(fmin(fmax(t1, t2), fmax(t3, t4)), fmax(t5, t6));

        if(tmax < 0) {return false;}
        if (tmin > 0)
            t = tmin;
        else
            t = tmax;
        return tmin <= tmax;
    }

    bool isIntersect(const Triangle& triangle) const {
        Vector3f v0 = triangle.vertices[0] - center;
        Vector3f v1 = triangle.vertices[1] - center;
        Vector3f v2 = triangle.vertices[2] - center;

        Vector3f edge0 = triangle.vertices[1] - triangle.vertices[0];
        Vector3f edge1 = triangle.vertices[2] - triangle.vertices[1];
        Vector3f edge2 = triangle.vertices[0] - triangle.vertices[2];

        return aabbTest(v0, v1, v2) &&
                planeTest(triangle.normal, v0) &&
               edgeTest(v0, v1, v2, edge0, edge1, edge2);
    }
    __host__ __device__ const Vector3f& getMin() const {
        return min_point;
    }
    __host__ __device__ const Vector3f& getMax() const {
        return max_point;
    }
    const Vector3f& getCenter() const {
        return center;
    }
private:
    bool edgeTest(const Vector3f& v0, const Vector3f& v1, const Vector3f& v2, const Vector3f& edge0, const Vector3f& edge1, const Vector3f& edge2) const {
        float fex = abs(edge0.x);
        float fey = abs(edge0.y);
        float fez = abs(edge0.z);
        if (!axisTestX(edge0.z, edge0.y, fez, fey, v0, v2)) {
            return false;
        }
        if (!axisTestY(edge0.z, edge0.x, fez, fex, v0, v2)) {
            return false;
        }
        if (!axisTestZ(edge0.y, edge0.x, fey, fex, v1, v2)) {
            return false;
        }

        fex = abs(edge1.x);
        fey = abs(edge1.y);
        fez = abs(edge1.z);
        if (!axisTestX(edge1.z, edge1.y, fez, fey, v0, v2)) {
            return false;
        }
        if (!axisTestY(edge1.z, edge1.x, fez, fex, v0, v2)) {
            return false;
        }
        if (!axisTestZ(edge1.y, edge1.x, fey, fex, v0, v1)) {
            return false;
        }

        fex = abs(edge2.x);
        fey = abs(edge2.y);
        fez = abs(edge2.z);
        if (!axisTestX(edge2.z, edge2.y, fez, fey, v0, v1)) {
            return false;
        }
        if (!axisTestY(edge2.z, edge2.x, fez, fex, v0, v1)) {
            return false;
        }
        if (!axisTestZ(edge2.y, edge2.x, fey, fex, v1, v2)) {
            return false;
        }
        return true;
    }
    bool aabbTest(const Vector3f& v0, const Vector3f& v1, const Vector3f& v2) const {
        float max = fmax(fmax(v0.x, v1.x), v2.x);
        float min = fmin(fmin(v0.x, v1.x), v2.x);
        if (min > boxHalfSize.x || max < -boxHalfSize.x) {
            return false;
        }

        max = fmax(fmax(v0.y, v1.y), v2.y);
        min = fmin(fmin(v0.y, v1.y), v2.y);
        if (min > boxHalfSize.y || max < -boxHalfSize.y) {
            return false;
        }

        max = fmax(fmax(v0.z, v1.z), v2.z);
        min = fmin(fmin(v0.z, v1.z), v2.z);
        return !(min > boxHalfSize.z || max < -boxHalfSize.z);
    }

    bool planeTest(const Vector3f& normal, const Vector3f& vertex) const {
        float vMax[3], vMin[3];
        float v;

        for (int dimension = 0; dimension < 3; dimension++) {
            v = getDimension(vertex, dimension);
            if (getDimension(normal, dimension) > 0) {
                vMin[dimension] = -getDimension(boxHalfSize, dimension) - v;
                vMax[dimension] = getDimension(boxHalfSize, dimension) - v;
            } else {
                vMin[dimension] = getDimension(boxHalfSize, dimension) - v;
                vMax[dimension] = -getDimension(boxHalfSize, dimension) - v;
            }
        }

        Vector3f max = make_float3(vMax[0], vMax[1], vMax[2]);
        Vector3f min = make_float3(vMin[0], vMin[1], vMin[2]);

        if (dot(normal, min) > 0) {
            return false;
        }
        if (dot(normal, max) >= 0) {
            return true;
        }
        return false;
    }

    bool axisTestX(const float a, const float b, const float fa, const float fb, const Vector3f& v0, const Vector3f& v1) const {
        float p0 = a*v0.y - b*v0.z;
        float p2 = a*v1.y - b*v1.z;

        float min = fmin(p0, p2);
        float max = fmax(p0, p2);

        float rad = fa * boxHalfSize.y + fb * boxHalfSize.z;

        return !(min > rad || max < -rad);
    }
    bool axisTestY(const float a, const float b, const float fa, const float fb, const Vector3f& v0, const Vector3f& v1) const {
        float p0 = a*v0.x - b*v0.z;
        float p2 = a*v1.x - b*v1.z;

        float min = fmin(p0, p2);
        float max = fmax(p0, p2);

        float rad = fa * boxHalfSize.x + fb * boxHalfSize.z;

        return !(min > rad || max < -rad);
    }
    bool axisTestZ(const float a, const float b, const float fa, const float fb, const Vector3f& v0, const Vector3f& v1) const {
        float p0 = a*v0.x - b*v0.y;
        float p2 = a*v1.x - b*v1.y;

        float min = fmin(p0, p2);
        float max = fmax(p0, p2);

        float rad = fa * boxHalfSize.x + fb * boxHalfSize.y;

        return !(min > rad || max < -rad);
    }
};


#endif //CUDA_TEST_AABB_CUH

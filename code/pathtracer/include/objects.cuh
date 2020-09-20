//
// Created by blackcloud on 2020/5/6.
//

#ifndef CUDA_TEST_OBJECTS_CUH
#define CUDA_TEST_OBJECTS_CUH
#include "sphere.cuh"
#include "plane.cuh"
#include "mesh.cuh"
#include "transform.cuh"
#include "bvh.cuh"
#include "revsurface.cuh"

__global__ void CreateSphere(float3 center, float radius, Material* mat, int num, Object3D** objects, AABB* aabb) {
    objects[num] = new Sphere(center, radius, mat);
    //printf("Num: %d Pointer: %d\n", num, objects[num]);
    *aabb = AABB(*objects[num]->getBox());
}
__global__ void CreatePlane(float3 norm, float d, float3 minp, float3 maxp, Material* mat, int num, Object3D** objects, AABB* aabb) {
    objects[num] = new Plane(norm, d, minp, maxp, mat);
    //printf("Num: %d Pointer: %d\n", num, objects[num]);
    *aabb = AABB(*objects[num]->getBox());
}
__global__ void CreatePlane(float3 norm, float d, const int range, Vector3f centerp, Vector3f texDirU, Vector3f texDirV, Material* mat, int num, Object3D** objects, AABB* aabb) {
    objects[num] = new Plane(norm, d, range, centerp, texDirU, texDirV, mat);
    //printf("Num: %d Pointer: %d\n", num, objects[num]);
    *aabb = AABB(*objects[num]->getBox());
}
__global__ void CreateMesh(KdTree* kdtree, Material* mat, int num, Object3D** objects, AABB* aabb) {
    //printf("Num: %d\n", num);
    objects[num] = new Mesh(kdtree, mat);
    *aabb = AABB(*objects[num]->getBox());
}
__global__ void CreateTransformedMesh(KdTree* kdtree, Material* mat, Matrix4f transfrom, int num, Object3D** objects, AABB* aabb) {
    //printf("Num: %d\n", num);
    objects[num] = new Transform(transfrom, new Mesh(kdtree, mat));
    *aabb = AABB(*objects[num]->getBox());
}
__global__ void CreateTransformedRevsurface(Vector3f* points, int n, int k, Material* mat, Matrix4f transfrom, int num, Object3D** objects, AABB* aabb, int curveflag) {
    //printf("Num: %d\n", num);
    if (curveflag == 0)
        objects[num] = new Transform(transfrom, new RevSurface(new BsplineCurve(points, n, k), mat));
    else
        objects[num] = new Transform(transfrom, new RevSurface(new BezierCurve(points, n, k), mat));
    *aabb = AABB(*objects[num]->getBox());
    //*aabb = AABB(transformPoint(transfrom, min), transformPoint(transfrom,max));
}

__global__ void DeleteObject(int num, Object3D** objects) {
    for (int i = 0; i < num; i++) {
        //printf("Deleting %d\n",objects[i]);
        delete objects[i]; //delete * here
    }
}

class Objects : public Managed {
public:
    Object3D **objects; // objects*[max_num], stored on device
    AABB *aabb_of_objects;
    int num = 0; //
    int max_num = 0; //
    BvhTree* bvhtree = nullptr;
    ThreadResource resource;

    Objects(int _max_num) : max_num(_max_num) {
        cudaMalloc(&objects, sizeof(Objects*) * max_num); // object*[max_num] on device
        cudaMemset(objects, 0, sizeof(Objects*) * max_num);
        cudaMallocManaged(&aabb_of_objects, sizeof(AABB) * max_num);
    }
    Objects(const Objects& o) {
        num = o.num;
        max_num = o.max_num;
        cudaMalloc(&objects, sizeof(Objects*) * max_num); // no deep copy here!
        cudaMallocManaged(&aabb_of_objects, sizeof(AABB) * max_num);
    }
    ~Objects() {
        DeleteObject<<<1,1>>>(num, objects); // *
        cudaDeviceSynchronize();
        cudaFree(objects); // **
        cudaFree(aabb_of_objects);
        delete bvhtree;
    }

    void addSphere(float3 center, float radius, Material* mat) {
        assert(num+1 <= max_num);
        AABB* aabb_d;
        cudaMalloc(&aabb_d, sizeof(AABB));

        CreateSphere<<<1,1>>>(center, radius, mat, num, objects, aabb_d); // malloc * here

        AABB* aabb_h = (AABB*)malloc(sizeof(AABB));
        cudaMemcpy(aabb_h, aabb_d, sizeof(AABB), cudaMemcpyDeviceToHost);
        aabb_of_objects[num] = AABB(*aabb_h);
        cudaFree(aabb_d);
        free(aabb_h);
        num++;
    }
    void addPlane(float3 norm, float d, float3 minp, float3 maxp, Material* mat) {
        assert(num+1 <= max_num);
        AABB* aabb_d;
        cudaMalloc(&aabb_d, sizeof(AABB));

        CreatePlane<<<1,1>>>(norm, d, minp, maxp, mat, num, objects, aabb_d); // malloc * here

        AABB* aabb_h = (AABB*)malloc(sizeof(AABB));
        cudaMemcpy(aabb_h, aabb_d, sizeof(AABB), cudaMemcpyDeviceToHost);
        aabb_of_objects[num] = AABB(*aabb_h);
        cudaFree(aabb_d);
        free(aabb_h);
        num++;
    }

    void addPlane(float3 norm, float d, const int range, Vector3f centerp, Vector3f texDirU, Vector3f texDirV, Material* mat) {
        assert(num+1 <= max_num);
        AABB* aabb_d;
        cudaMalloc(&aabb_d, sizeof(AABB));

        CreatePlane<<<1,1>>>(norm, d, range, centerp, texDirU, texDirV, mat, num, objects, aabb_d); // malloc * here

        AABB* aabb_h = (AABB*)malloc(sizeof(AABB));
        cudaMemcpy(aabb_h, aabb_d, sizeof(AABB), cudaMemcpyDeviceToHost);
        aabb_of_objects[num] = AABB(*aabb_h);
        cudaFree(aabb_d);
        free(aabb_h);
        num++;
    }

    void addMesh(KdTree* kdtree, Material* mat) {
        assert(num+1 <= max_num);
        resource._mark_size = fmax(kdtree->getTriangleSize() / 8 + 1, resource._mark_size);
        AABB* aabb_d;
        cudaMalloc(&aabb_d, sizeof(AABB));

        CreateMesh<<<1,1>>>(kdtree, mat, num, objects, aabb_d);

        AABB* aabb_h = (AABB*)malloc(sizeof(AABB));
        cudaMemcpy(aabb_h, aabb_d, sizeof(AABB), cudaMemcpyDeviceToHost);
        aabb_of_objects[num] = AABB(*aabb_h);
        cudaFree(aabb_d);
        free(aabb_h);
        num++;
    }
    void addTransformedMesh(KdTree* kdtree, Material* mat, Matrix4f m) {
        assert(num+1 <= max_num);
        resource._mark_size = fmax(kdtree->getTriangleSize() / 8 + 1, resource._mark_size);
        AABB* aabb_d;
        cudaMalloc(&aabb_d, sizeof(AABB));

        CreateTransformedMesh<<<1,1>>>(kdtree, mat, m, num, objects, aabb_d);

        AABB* aabb_h = (AABB*)malloc(sizeof(AABB));
        cudaMemcpy(aabb_h, aabb_d, sizeof(AABB), cudaMemcpyDeviceToHost);
        aabb_of_objects[num] = AABB(*aabb_h);
        cudaFree(aabb_d);
        free(aabb_h);
        num++;
    }

    void addTransformedRevsurface(Vector3f* points, int n, int k, Material* mat, Matrix4f m, int curveflag) {
        assert(num+1 <= max_num);
        resource._base_size = fmax(n, resource._base_size);
        resource._derivative_size = fmax(n, resource._derivative_size);
        resource._s_size = fmax(k+2, resource._s_size);
        resource._ds_size = fmax(k+1, resource._ds_size);

        AABB* aabb_d;
        cudaMalloc(&aabb_d, sizeof(AABB));

        CreateTransformedRevsurface<<<1,1>>>(points, n, k, mat, m, num, objects, aabb_d, curveflag);

        AABB* aabb_h = (AABB*)malloc(sizeof(AABB));
        cudaMemcpy(aabb_h, aabb_d, sizeof(AABB), cudaMemcpyDeviceToHost);
        aabb_of_objects[num] = AABB(*aabb_h);
        cudaFree(aabb_d);
        free(aabb_h);
        num++;
    }






    void initBvh() {
        this->bvhtree = new BvhTree(objects, num, aabb_of_objects);
    }
    __device__ const ThreadResource& getResource() const {
        return resource;
    }
    __device__ bool intersect(const Ray &ray, Hit &hit, float tmin, ThreadResource* thread_resource) const {
        return bvhtree->intersect(ray,hit,tmin,thread_resource);
    }
};


#endif //CUDA_TEST_OBJECTS_CUH

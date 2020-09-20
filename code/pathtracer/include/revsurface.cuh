//
// Created by blackcloud on 2020/5/10.
//

#ifndef CUDA_TEST_REVSURFACE_CUH
#define CUDA_TEST_REVSURFACE_CUH
#include "object3d.cuh"
#include "curve.cuh"
#include <curand.h>
#include <curand_kernel.h>
#define GN_ITERATION_TIME_LIMIT 5
#define GN_TRY_TIME 30
#define GN_ITERATION_VALUE_LIMIT .0005f

//#define ERROR_EXIT -1
//
///* Check the return value of CUDA Runtime API */
//#define CHECK_CUDA(err) do{\
//    if((err) != cudaSuccess){\
//        fprintf(stderr, "CUDA Runtime API error %d at file %s line %d: %s.\n",\
//                               (int)(err), __FILE__, __LINE__, cudaGetErrorString((err)));\
//        exit(ERROR_EXIT);\
//    }}while(0)
//
///* Check the return value of CURAND api. */
//#define CHECK_CURAND(err) do{\
//    if( (err) != CURAND_STATUS_SUCCESS ){\
//        fprintf(stderr, "CURAND error %d at file %s line %d.\n", (int)(err), __FILE__, __LINE__);\
//	exit(ERROR_EXIT);\
//    }}while(0)
//
///* Function: produce random float data by GPU
// * Input:   dataHost: the memory to store data produced
// *          number: the number of data to produce
// *          seed: the seed for random generator
// * Output: void
// */
//extern "C"
//void randomGenerator(float *dataHost, int number, unsigned long long seed)
//{
//    float *dataDev;
//    CHECK_CUDA( cudaMalloc( (void **) &dataDev, number * sizeof(float) ) );
//
//    curandGenerator_t gen;
//    CHECK_CURAND( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
//    CHECK_CURAND( curandSetPseudoRandomGeneratorSeed(gen, seed) );
//    CHECK_CURAND( curandGenerateUniform(gen, dataDev, number) );
//    CHECK_CURAND( curandDestroyGenerator(gen) );
//
//    CHECK_CUDA( cudaMemcpy(dataHost, dataDev, number * sizeof(float), cudaMemcpyDeviceToHost) );
//    CHECK_CUDA( cudaFree(dataDev) );
//
//    return;
//}
class RevSurface : public Object3D {
    Curve *pCurve;
public:
    inline __device__ Vector3f getF(const Ray& r, float t, const Vector3f& V, const Vector3f& T, float theta) const {
        Vector3f l = r.pointAtParameter(t);
        Vector3f s = make_float3(V.x * cos(theta), V.y, V.x * sin(theta));
        return l - s;
    }
    inline __device__ Matrix3f getDF(const Ray& r, float t, const Vector3f& V, const Vector3f& T, float theta) const {
        const Vector3f& dir = r.getDirection();
        return Matrix3f(
                dir.x, -T.x * cos(theta), V.x * sin(theta),
                dir.y, -T.y, 0,
                dir.z, -T.x * sin(theta), -V.x * cos(theta)
                );
    }
    __device__ RevSurface(Curve *pCurve, Material* material) : pCurve(pCurve), Object3D(material) {
        // Check flat.
        auto controls = pCurve->getControls();
        auto controls_size = pCurve->getControlsSize();
        float maxX, minY, maxY;
        maxX = abs(controls[0].x);
        minY = controls[0].y;
        maxY = controls[0].y;

        printf("Revsurface, controls num: %d\n", controls_size);

        for (int i = 0; i < controls_size; i++) {
            if (controls[i].z != 0.0) {
                printf("Profile of revSurface must be flat on xy plane.\n");
                //exit(0);
            }

            minY = minY < controls[i].y ? minY : controls[i].y;
            maxX = maxX > abs(controls[i].x) ? maxX : abs(controls[i].x);
            maxY = maxY > controls[i].y ? maxY : controls[i].y;
        }
        printf("Revsurface, MaxX %f, MaxY %f, MinY %f\n",maxX, maxY, minY);
        box = new AABB(make_float3(-maxX-1, minY-1, -maxX-1), make_float3(maxX+1, maxY+1, maxX+1));



        //debug

//        for (int i = 0; i < 100; i++) {
//            Vector3f p,t;
//            pCurve->getBaseAndDerivativeAtMu(double(i)/double(100), &p, &t, NULL);
//            printVector(p);
//            printVector(t);
//            printf("\n");
//        }
    }

    __device__ ~RevSurface() override {
        //delete box;
        delete pCurve;
    }

    __device__ bool intersect(const Ray &r, Hit &h, float tmin, ThreadResource* thread_resource) const override {
        float tt = 0;
        if (this->box->isIntersect(r, tt)) {
            //uint seed[] = {uint(rand()), uint(rand())};
            curandState_t state;
            curand_init(clock64(), 0,0,&state);
            float currt = INFINITY;
            float curru;
            float currtheta;
            Vector3f currV;
            Vector3f currT;
            for (int i = 0; i < GN_TRY_TIME; i++) {
                float t = tt;
                //float u = rand(seed); // rand float within (0,1)
                //float u = float(i)/GN_TRY_TIME;
                //float theta = rand(seed) * M_PI * 2;
                float u = curand_uniform(&state);
                float theta = curand_uniform(&state) * M_PI * 2;
                Vector3f params = make_float3(t, u, theta);
                bool flag = false;
                Vector3f V,T;

                for (int iter = 0; iter < GN_ITERATION_TIME_LIMIT; iter++) {
                    t = params.x;
                    u = params.y;
                    theta = params.z;
                    //printf("Iter %d param %f %f %f\n",iter,params.x, params.y, params.z);
                    //printVector(params);
                    if (u > 1.5f || u < -0.5f) {
                        break;
                    }

                    pCurve->getBaseAndDerivativeAtMu(u, &V, &T, thread_resource);
                    auto F = getF(r, t, V, T, theta);
                    auto DF = getDF(r, t, V, T, theta);
                    if (max(max(abs(F.x), abs(F.y)), abs(F.z)) < GN_ITERATION_VALUE_LIMIT) {
                        flag = true;
                        break;
                    }

                    params -= DF.inverse() * F;

                }
                if (!flag || u < 0 || u > 1)
                    continue;
                if (currt > t && t > tmin) {
                    currt = t;
                    curru = u;
                    currtheta = theta;
                    currV = V;
                    currT = T;
                }
            }

            if (currt < h.getT()) {
                Vector3f n = make_float3(-currT.y, currT.x, 0);
                if (dot(currV, n) < 0)
                    n *= -1;
                Vector3f nn = normalize(make_float3(n.x * cos(currtheta), n.y, n.x * sin(currtheta)));
                h.set(currt, material->getMaterialFeatureAtPoint(0,0), nn);
                return true;
            }
        }
        return false;
    }
};

#endif //CUDA_TEST_REVSURFACE_CUH

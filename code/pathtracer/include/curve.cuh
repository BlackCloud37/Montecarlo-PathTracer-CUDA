//
// Created by blackcloud on 2020/5/10.
//

#ifndef CUDA_TEST_CURVE_CUH
#define CUDA_TEST_CURVE_CUH
#include "object3d.cuh"
#include "vec/vec_mat.cuh"

// The CurvePoint object stores information about a point on a curve
// after it has been tesselated: the vertex (V) and the tangent (T)
// It is the responsiblility of functions that create these objects to fill in all the data.
#define printVector(vec) do {                                          \
    printf("%f %f %f\n", vec.x, vec.y, vec.z);                           \
} while(0)

class Curve {
protected:
    int n; // num of controls
    int k; // degrees
    Vector3f* controls;
public:
    __device__ explicit Curve(Vector3f* points, int n, int k) : n(n), k(k) {
        controls = (Vector3f*)malloc(sizeof(Vector3f) * n);
        for (int i = 0; i < n; i++) {
            controls[i] = points[i];
        }
    }

    __device__ virtual  ~Curve() {
        free(controls);
    }
    __device__ Vector3f *getControls() const {
        return controls;
    }
    __device__ int getControlsSize() const {
        return n;
    }
    __device__ virtual float getMinMu() const = 0;
    __device__ virtual float getMaxMu() const = 0;
    __device__ virtual void getBaseAndDerivativeAtMu(float mu, Vector3f* point, Vector3f* tang, ThreadResource *resource) const = 0;
};

struct CurvePoint {
    Vector3f V; // Vertex
    Vector3f T; // Tangent  (unit)
    CurvePoint() {}
    CurvePoint(Vector3f _V, Vector3f _T) : V(_V), T(_T) {}
};

class BsplineCurve : public Curve {

    __device__ static int upper_bound(const int begin, const int length, const float* arr, const float key) {
        int mid, first = begin, last = length - 1;
        while (first <= last) {
            mid = last - (last - first) / 2;
            if (arr[mid] <= key) first = mid + 1;
            else last = mid - 1;
        }
        return first;
    }
    __device__ static int lower_bound(const int begin, const int length, const float* arr, const float key) {
        int mid, first = begin, last = length - 1;
        while (first <= last) {
            mid = last - (last - first) / 2;
            if (arr[mid] >= key) last = mid - 1;
            else first = mid + 1;
        }
        return first;
    }
protected:
    __device__ int get_bpos(const float mu) const {
        int bpos;
        if (mu < knot[0] || mu > knot[knot_size-1])
            return -1;
        if (mu == knot[0])
            bpos = upper_bound(0, knot_size, knot, mu) - 1;
        else
            bpos = max(0, lower_bound(0, knot_size, knot, mu) - 1);
        return bpos;
    }

    int knot_size;
    float* knot;
    int tpad_size;
    float* tpad;
public:
    __device__ float getMinMu() const override {
        return knot[k];
    }
    __device__ float getMaxMu() const override {
        return knot[n];
    }
    __device__ explicit BsplineCurve(Vector3f* points, int n, int k) : Curve(points, n, k) {
        if (n < 4) {
            printf("Number of control points of BspineCurve must be more than 4!\n");
            return;
        }
        knot_size = n + k + 1;
        knot = (float*)malloc(sizeof(float) * knot_size);

        tpad_size = knot_size + k;
        tpad = (float*)malloc(sizeof(float) * tpad_size);
        for (int i = 0; i < knot_size; i++) {
            knot[i] = float(i)/float(knot_size-1);
            tpad[i] = knot[i];
        }
        for (int i = knot_size; i < knot_size + k; i++)
            tpad[i] = knot[knot_size - 1];
        printf("n: %d\nk: %d\n", n, k);
        printf("Knot:\n");
        for (int i = 0; i < knot_size; i++)
            printf("%f ", knot[i]);
        printf("\nTpad:\n");
        for (int i = 0; i < tpad_size; i++)
            printf("%f ", tpad[i]);
        printf("\n");
    }
    __device__ ~BsplineCurve() override {
        free(controls);
        free(knot);
        free(tpad);
    }

    __device__ void getBaseAndDerivativeAtMu(float mu, Vector3f* point, Vector3f* tang, ThreadResource *resource) const final {
        //printf("Called\n");
        if (mu < 0)
            mu = 0.0001;
        if (mu > 1)
            mu = 0.9999;
        auto bpos = get_bpos(mu);
        float s[100]; memset(s, 0, sizeof(float) * (k+2));
        int s_end = k+2;
        int s_begin = 0;
        s[k] = 1;

        float ds[100];
        for (int i = 0; i < k+1; i++) ds[i] = 1;
        int ds_end = k+1;
        int ds_begin = 0;

        for (int p = 1; p < k+1; p++) {
            for (int ii = k - p; ii < k+1; ii++) {
                auto i = ii + bpos - k;
                float w1,dw1,w2,dw2;

                if (tpad[i + p] == tpad[i]) {
                    w1 = mu;
                    dw1 = 1.;
                }
                else {
                    w1 = (mu - tpad[i]) / (tpad[i + p] - tpad[i]);
                    dw1 = 1.f / (tpad[i + p] - tpad[i]);
                }
                if (tpad[i + p + 1] == tpad[i + 1]) {
                    w2 = 1.f - mu;
                    dw2 = -1.;
                }
                else {
                    w2 = (tpad[i + p + 1] - mu) / (tpad[i + p + 1] - tpad[i + 1]);
                    dw2 = -1.f / (tpad[i + p + 1] - tpad[i + 1]);
                }
                if (p == k)
                    ds[ii] = (dw1 * s[ii] + dw2 * s[ii + 1]) * float(p);
                s[ii] = w1 * s[ii] + w2 * s[ii + 1];
            }
        }

        s_end--; // s.pop_back
        auto lsk = bpos - k;
        auto rsk = n - bpos - 1;

        if (lsk < 0) {
            s_begin -= lsk;
            ds_begin -= lsk;
            lsk = 0;
        }
        if (rsk < 0) {
            s_end += rsk;
            ds_end += rsk;
        }
        Vector3f t = make_float3(0,0,0), p = make_float3(0,0,0);

        float s_ret[100], ds_ret[100];
        memset(s_ret, 0, sizeof(float) * 100);
        memset(ds_ret, 0, sizeof(float)* 100);
        memcpy(s_ret+lsk,s+s_begin,(s_end-s_begin)*sizeof(float));
        memcpy(ds_ret+lsk,ds+ds_begin,(ds_end-ds_begin)*sizeof(float));

//        for (int i = 0; i < n; i++) {
//            printf("%f ",s_ret[i]);
//        }
//        printf("\n");
//        for (int i = 0; i < n; i++) {
//            printf("%f ",ds_ret[i]);
//        }
//        printf("\n");
        for (int i = 0; i < n; i++) {
            p += controls[i] * s_ret[i];
            t += controls[i] * ds_ret[i];
        }
//        for (int i = s_begin; i < s_end; i++) {
//            p += controls[i] * s[i];
//            t += controls[i] * ds[i];
//        }
        *point = p;
        *tang = t;
    }
};
class BezierCurve : public BsplineCurve {
public:
    __device__ virtual float getMinMu() const override {
        //TODO:
        return 0.5f;
    }
    __device__ explicit BezierCurve(Vector3f* points, int n, int k) : BsplineCurve(points, n, k) {
        if (n < 4) {
            printf("Number of control points of BspineCurve must be more than 4!\n");
            return;
        }
        n = getControlsSize();
        k = n-1;
        free(knot);
        free(tpad);

        knot_size = n + k + 1;
        knot = (float*)malloc(sizeof(float) * knot_size);
        for (int i = 0; i < n; i++)
            knot[i] = 0;
        for (int i = n; i < knot_size; i++)
            knot[i] = 1.;

        tpad_size = knot_size + k;
        tpad = (float*)malloc(sizeof(float) * tpad_size);
        memcpy(tpad, knot, 2*n*sizeof(float));
        for (int i = 0; i < k; i++) tpad[knot_size + i] = knot[knot_size-1];
        printf("n: %d\nk: %d\n", n, k);
        printf("Knot:\n");
        for (int i = 0; i < knot_size; i++)
            printf("%f ", knot[i]);
        printf("\nTpad:\n");
        for (int i = 0; i < tpad_size; i++)
            printf("%f ", tpad[i]);
        printf("\n");
    }
};
#endif //CUDA_TEST_CURVE_CUH

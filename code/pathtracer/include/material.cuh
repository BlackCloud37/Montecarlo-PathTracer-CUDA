//
// Created by blackcloud on 2020/5/5.
//

#ifndef CUDA_TEST_MATERIAL_CUH
#define CUDA_TEST_MATERIAL_CUH
#include <cassert>

#include "ray.cuh"
#include "managed.cuh"
#include <iostream>

enum REFL_TYPE {
    DIFF,
    SPEC,
    REFR
};
//texture<Vector3f, cudaTextureType2D, cudaReadModeNormalizedFloat> emRefTex;
//texture<REFL_TYPE, cudaTextureType2D, cudaReadModeNormalizedFloat> typeRefTex;
//texture<float, cudaTextureType2D, cudaReadModeNormalizedFloat> bumpRefTex;
struct MaterialFeature {
    Vector3f emission;
    Vector3f color;
    REFL_TYPE type;
    float gradientU, gradientV;
    __device__ MaterialFeature() = default;
    __device__ MaterialFeature(const Vector3f& e, const Vector3f& c, const REFL_TYPE t, const float gu, const float gv) {
        emission = e;
        color = c;
        type = t;
        gradientU = gu;
        gradientV = gv;
    }
    __device__ MaterialFeature(const MaterialFeature& mf) {
        emission = mf.emission;
        color = mf.color;
        type = mf.type;
        gradientU = mf.gradientU;
        gradientV = mf.gradientV;
    }
    __device__ const Vector3f& getColor() const {
        return color;
    }
    __device__ const Vector3f& getEmission() const {
        return emission;
    }
    __device__ const REFL_TYPE& getType() const {
        return type;
    }
};

class Texture : public Managed {
    Vector3f* emission_map;
    Vector3f* color_map;
    REFL_TYPE* type_map;
    Vector3f* bump_map;
    int width;
    int height;
//    float realW;
//    float realH;
public:
    Texture(const char* _emf, const char* _cmf, const char* _tmf, const char* _bmf, const float _rw = 0, const float _rh = 0) {
        //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        emission_map = nullptr;
        color_map = nullptr;
        type_map = nullptr;
        bump_map = nullptr;

        if (_emf != nullptr) {
            auto _em = Image::loadPPM(_emf, width, height);
            cudaMalloc(&emission_map, sizeof(Vector3f) * width * height);
            assert(emission_map != nullptr);
            cudaMemcpy(emission_map, _em, sizeof(Vector3f) * width * height, cudaMemcpyHostToDevice);
            delete[] _em;
        }

        if (_cmf != nullptr) {
            auto _cm = Image::loadPPM(_cmf, width, height);
            cudaMalloc(&color_map, sizeof(Vector3f) * width * height);
            assert(color_map != nullptr);
            cudaMemcpy(color_map, _cm, sizeof(Vector3f) * width * height, cudaMemcpyHostToDevice);
            delete[] _cm;
        }

        if (_tmf != nullptr) {
            auto _tm = Image::loadPPM(_tmf, width, height);
            cudaMalloc(&type_map, sizeof(REFL_TYPE) * width * height);
            assert(type_map != nullptr);
            auto _tmT = new REFL_TYPE[width * height];
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    int index = Image::index(x, y, width, height);
                    int max_idx = (_tm[index].x >= _tm[index].y && _tm[index].x >= _tm[index].z) ? 0 : (_tm[index].y >= _tm[index].z && _tm[index].y >= _tm[index].x ? 1 : 2);
                    if (max_idx < 0 || max_idx > 2) printf("Type num err\n");
                    _tmT[index] = max_idx == 0 ? DIFF : (max_idx == 1 ? SPEC : REFR);
                }
            }
            cudaMemcpy(type_map, _tmT, sizeof(REFL_TYPE) * width * height, cudaMemcpyHostToDevice);
            delete[] _tmT;
            delete[] _tm;
        }

        if (_bmf != nullptr) {
            auto _bm = Image::loadPPM(_bmf, width, height);
            // TODO: gradient in y & z
            float max_gu = 0, max_gv = 0;
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    int prev_x = x == 0 ? x : x-1;
                    int prev_y = y == 0 ? y : y-1;
                    int next_x = x == width-1 ? x : x+1;
                    int next_y = y == height-1 ? y : y+1;

                    int index_l = Image::index(prev_x, y, width, height);
                    int index_r = Image::index(next_x, y, width, height);
                    int index_u = Image::index(x, prev_y, width, height);
                    int index_d = Image::index(x, next_y, width, height);
                    int index = Image::index(x,y,width,height);

                    _bm[index].y = (_bm[index_r].x - _bm[index_l].x) * (index_r == index || index_l == index ? 1.f : .5f);// gu hor
                    max_gu = fmax(_bm[index].y, max_gu);
                    _bm[index].y = Image::clamp(_bm[index].y * 10);
                    _bm[index].z = (_bm[index_u].x - _bm[index_d].x) * (index_u == index || index_d == index ? 1.f : .5f);// gv vert
                    _bm[index].z = Image::clamp(_bm[index].z * 10);
                    max_gv = fmax(_bm[index].z, max_gv);
                    //printf("x,y %d,%d, gu %f gv %f\n",x,y,_bm[index].y,_bm[index].z);

                }
            }

//            for (int x = 0; x < width; x++) {
//                for (int y = 0; y < width; y++) {
//                    int index = Image::index(x,y,width,height);
//                    _bm[index].y /= max_gu;
//                    _bm[index].z /= max_gv;
//                    printf("x,y %d,%d, gu %f gv %f\n",x,y,_bm[index].y,_bm[index].z);
//                }
//            }

            cudaMalloc(&bump_map, sizeof(Vector3f) * width * height);
            assert(bump_map != nullptr);
            cudaMemcpy(bump_map, _bm, sizeof(Vector3f) * width * height, cudaMemcpyHostToDevice);
            delete[] _bm;
        }
//        realW = _rw == 0 ? width : _rw;
//        realH = _rh == 0 ? height : _rh;
    }
    ~Texture() {
        if (emission_map)
            cudaFree(emission_map);
        if (color_map)
            cudaFree(color_map);
        if (type_map)
            cudaFree(type_map);
        if (bump_map)
            cudaFree(bump_map);
    }
    __device__ void getMaterialAt(const float u, const float v, MaterialFeature& ft) {
        if (u < 0 || u > 1 || v < 0 || v > 1) {
            printf("Incorrect u,v: %f %f\n", u, v);
            return;
        }
        int mapped_x = (int)(u * width);
        int mapped_y = (int)(v * height);
        int index = Image::index(mapped_x, mapped_y, width, height);
        if (emission_map) {
            ft.emission = emission_map[index];
        }
        if (color_map) {
            ft.color = color_map[index];
        }
        if (type_map) {
            ft.type = type_map[index];
        }
        if (bump_map) {
            ft.gradientU = bump_map[index].y;
            ft.gradientV = bump_map[index].z;
        }
    }
};



// Implement Shade function that computes Phong introduced in class.
class Material : public Managed{
    REFL_TYPE refl;
    Vector3f emission, color;
    Texture* texture;
public:
    Material() = default;
    Material(const Material& m) {
        texture = m.texture;
        refl = m.refl;
        emission = m.emission;
        color = m.color;
    }
    __host__ Material(const Vector3f& _e, const Vector3f& _c, const REFL_TYPE _t, Texture* _tex = nullptr) :
            emission(_e), color(_c), refl(_t), texture(_tex) {};
    ~Material() {
        delete texture;
    }
    __device__ const Vector3f& getColor() const {
        return color;
    }
    __device__ const Vector3f& getEmission() const {
        return emission;
    }
    __device__ const REFL_TYPE& getType() const {
        return refl;
    }
    __device__ MaterialFeature getMaterialFeatureAtPoint(float u, float v) const {
        MaterialFeature mf(emission, color, refl, 0, 0);
        //printf("Type: %d\n", refl);
        if (texture)
            texture->getMaterialAt(u, v, mf);
//        printf("Type: %d\n", mf.type);
//        printf("Emission: %f %f %f \n", mf.emission.x, mf.emission.y, mf.emission.z);
        return mf;
    }
    __device__ bool ifHasTexture() const {
        return texture != nullptr;
    }
};
#endif //CUDA_TEST_MATERIAL_CUH

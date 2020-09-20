//
// Created by blackcloud on 2020/5/5.
//

#ifndef CUDA_TEST_IMAGE_CUH
#define CUDA_TEST_IMAGE_CUH

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include "vec/vec_mat.cuh"


class Image {
    /* Color is in [0,1] */
    Vector3f* data;
    int width, height;

public:
    inline int index(int x, int y) const {
        return (height - y - 1) * width + x;
    }
    __device__ __host__ static inline int index(int x, int y, int w, int h) {
        int res = (h - y - 1) * w + x;
        if (res < 0 || res >= w * h) {
            printf("Error at index()\n");
            return 0;
        }
        return res;
    }
    Image() = delete ;

    Image(const int _w, const int _h):width(_w),height(_h) {
        data = new Vector3f[width * height];
    }
    ~Image() {
        delete[] data;
    }

    void setPixel(const int x, const int y, const Vector3f& color) {
        assert(x >= 0 && x < width);
        assert(y >= 0 && y < height);
        int idx = index(x, y);
        data[idx] = color;
    }

    void savePPM(const char* output_filename) const {
        char filename[256] = {0};
        if (!strcmp(output_filename, "")) {
            //产生文件名: 月份-日期_小时-秒-ssp
            time_t t;
            struct tm *p;
            t = time(nullptr);
            p = gmtime(&t);
            sprintf(filename, "result/image%d_%d-%d_%d.ppm", 1 + p->tm_mon, p->tm_mday, p->tm_hour + 8, p->tm_min);
        }
        else {
            strcpy(filename, output_filename);
        }

        FILE *f = std::fopen(filename, "w");
        fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
        for (int i = 0; i < width * height; i++)
            fprintf(f, "%d %d %d ", toInt(data[i].x), toInt(data[i].y), toInt(data[i].z));
    }
    static void savePPM(const char* output_filename, Vector3f* i_data, const int width, const int height) {
        char filename[256] = {0};
        if (!strcmp(output_filename, "")) {
            //产生文件名: 月份-日期_小时-秒-ssp
            time_t t;
            struct tm *p;
            t = time(nullptr);
            p = gmtime(&t);
            sprintf(filename, "result/image%d_%d-%d_%d.ppm", 1 + p->tm_mon, p->tm_mday, p->tm_hour + 8, p->tm_min);
        }
        else {
            strcpy(filename, output_filename);
        }

        FILE *f = std::fopen(filename, "w");
        fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
        for (int i = 0; i < width * height; i++)
            fprintf(f, "%d %d %d ", toInt(i_data[i].x), toInt(i_data[i].y), toInt(i_data[i].z));
    }
    __device__ __host__ static inline float clamp(float x){ return x<0 ? 0 : x>1 ? 1 : x; };
    __device__ __host__ static inline int toInt(const float x){ return int(pow(clamp(x),1/2.2)*255+.5); }

    __host__ static Vector3f* loadPPM(const char* filename, int& w, int& h) {
        assert(filename != nullptr);
        // must end in .ppm
        const char *ext = &filename[strlen(filename)-4];
        assert(!strcmp(ext,".ppm"));
        FILE *file = fopen(filename,"rb");
        // misc header information
        int width = 0;
        int height = 0;
        char tmp[100];
        fgets(tmp,100,file);
        assert (strstr(tmp,"P6"));
        fgets(tmp,100,file);
        if (tmp[0] == '#')
            fgets(tmp,100,file);
        // assert (tmp[0] == '#');
        // fgets(tmp,100,file);
        sscanf(tmp,"%d %d",&width,&height);
        fgets(tmp,100,file);
        assert (strstr(tmp,"255"));
        // the data
        Vector3f* data = new Vector3f[width * height];
        assert(data != nullptr);
        // flip y so that (0,0) is bottom left corner
        for (int y = height-1; y >= 0; y--) {
            for (int x = 0; x < width; x++) {
                unsigned char r,g,b;
                r = fgetc(file);
                g = fgetc(file);
                b = fgetc(file);
                data[index(x,y,width,height)] = make_float3(r/255.0,g/255.0,b/255.0);
            }
        }
        fclose(file);
        w = width;
        h = height;
        return data;
    }
};

#endif //CUDA_TEST_IMAGE_CUH

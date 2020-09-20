//
// Created by blackcloud on 2020/5/16.
//

#ifndef CUDA_TEST_PATHTRACER_CUH
#define CUDA_TEST_PATHTRACER_CUH
#include "camera.cuh"


#define MAX_RECURSIVE_DEPTH 8
#define ABSORB_COE_OF_MEDIA 0.0000f
#define SCATTER_COE_OF_MEDIA 0.0002f
#define SCATTER_COE_OF_MEDIA_FRAC 5000
//#define ABSORB_SCATTER_COE_OF_MEDIA 0.01f
#define RATIO_A_AS .0f









template<int depth>
__device__ Vector3f radiance(const Ray& r, const Objects* scene, uint* Xi, ThreadResource* resource) {
    /* Intersect test */
    Hit h;
    if (!scene->intersect(r, h, 0, resource))
        return make_float3(0,0,0);

    /* If intersect */
    const float t = h.getT();

//    /* test absorb or scatter */
//    if (depth <= 1 && rand(Xi) < 1 - (__expf(-t * (SCATTER_COE_OF_MEDIA + ABSORB_COE_OF_MEDIA)))) {
//        /* absorbed or scattered */
//        if (rand(Xi) < RATIO_A_AS) {
//            /* absorbed */
//            return make_float3(0,0,0);
//        }
//        else {
//            /* simulate scatter, choose a point at the ray as the scatter point */
//            float scatter_t = -__logf(1 - (1 - __expf(-t * SCATTER_COE_OF_MEDIA)) * rand(Xi) ) * SCATTER_COE_OF_MEDIA_FRAC;
//            float u = 2.f * M_PI * rand(Xi), v = M_PI * rand(Xi);
//            Vector3f scatter_point = r.pointAtParameter(scatter_t);
//            Vector3f scatter_dir = make_float3(__sinf(u) * __cosf(v), __sinf(u) * __sinf(v), __cosf(u));
//            return mult(make_float3(1.f, 1.f, 1.f), radiance<depth+1>(Ray(scatter_point, scatter_dir), scene, Xi, resource));
//        }
//    }

    /* Not absorbed or scattered */
    //const Material* material = h.getMaterial();
    const auto mf = h.getFeature();
    const auto *material = &mf;
    Vector3f hit_point = r.pointAtParameter(t),
            n = h.getNormal(),
            nl = dot(n, r.getDirection()) < 0 ? n : -1*n,
            hit_color = material->getColor();

    float p = hit_color.x > hit_color.y && hit_color.x > hit_color.z ? hit_color.x : hit_color.y > hit_color.z ? hit_color.y : hit_color.z;
    if (depth+1 > MAX_RECURSIVE_DEPTH) {
        if (rand(Xi) < p) {
            hit_color = hit_color * (1 / p);
        }
        else {
            return material->getEmission();
        }
    }


    if (material->getType() == DIFF) {
        float r1 = 2.f * M_PI * rand(Xi), r2 = rand(Xi), r2s = sqrt(r2);
        Vector3f w = nl,
                u = normalize(cross((fabs(w.x) > .1f ? make_float3(0 , 1, 0) : make_float3(1, 0, 0)), w)),
                v = cross(w, u);
        Vector3f d = normalize((u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1-r2)));
        return material->getEmission() + mult(hit_color, radiance<depth+1>(Ray(hit_point, d), scene, Xi, resource));
    }
    else if (material->getType() == SPEC) {
        return material->getEmission() + mult(hit_color, radiance<depth+1>(Ray(hit_point, r.getDirection() - n * 2 * dot(n, r.getDirection())), scene, Xi, resource));
    }
    else {
        Ray reflRay(hit_point, r.getDirection() - n * 2 * dot(n, r.getDirection()));
        bool into = dot(n, nl) > 0;
        float nc = 1,
                nt = 1.5,
                nnt = into ? nc / nt : nt / nc,
                ddn = dot(r.getDirection(), nl),
                cos2t;


        if ((cos2t = 1 - nnt*nnt*(1 - ddn*ddn)) < 0)
            return material->getEmission() + mult(hit_color, radiance<depth+1>(reflRay, scene, Xi, resource));


        Vector3f tdir = normalize((r.getDirection() * nnt - n * ((into ? 1:-1) * (ddn * nnt + sqrt(cos2t)))));
        float a = nt - nc,
                b = nt + nc,
                R0 = a*a/(b*b),
                c = 1-(into ? -ddn:dot(tdir, n));
        float Re = R0 + (1 - R0) * c * c * c * c * c,
                Tr = 1 - Re,
                P = 0.25f + 0.5f * Re,
                RP = Re / P,
                TP = Tr / (1-P);
        return material->getEmission() + mult(hit_color,
                                              depth > 2 ?
                                              (rand(Xi) < P ? radiance<depth+1>(reflRay, scene, Xi, resource) * RP : radiance<depth+1>(Ray(hit_point, tdir), scene, Xi, resource) * TP) :
                                              radiance<depth+1>(reflRay, scene, Xi, resource) * Re + radiance<depth+1>(Ray(hit_point, tdir), scene, Xi, resource) * Tr);
    }
}

template<>
__device__ Vector3f radiance<MAX_RECURSIVE_DEPTH> (const Ray& ray, const Objects* scene, uint *Xi, ThreadResource* resource) {
    return make_float3(0,0,0);
}


#endif //CUDA_TEST_PATHTRACER_CUH

cmake_minimum_required(VERSION 3.16)
project(pathtracer CUDA)

set(CMAKE_CUDA_STANDARD 14)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
add_executable(pathtracer
        main.cu
        include/image.cuh
        #include/vec/Vector3f.cuh
        #include/vec/Vector3f.cu
        #include/vec/Vector2f.cuh
        #include/vec/Vecor2f.cu
        include/camera.cuh
        include/ray.cuh
        include/object3d.cuh
        include/hit.cuh
        include/material.cuh
        include/utils.cuh
        include/sphere.cuh
        include/managed.cuh
        include/objects.cuh
        include/plane.cuh
        #include/scene.cuh
        /usr/local/cuda/samples/common/inc/helper_math.h include/aabb.cuh include/vec/helper_math.h include/kdtree.cuh include/triangle.cuh include/mesh.cuh include/transform.cuh include/vec/vec_mat.cuh include/threadresource.cuh include/bvh.cuh include/sceneparser.cuh include/curve.cuh include/revsurface.cuh include/pathtracer.cuh)
set_target_properties(
        pathtracer
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

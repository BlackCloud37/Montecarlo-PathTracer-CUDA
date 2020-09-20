//
// Created by blackcloud on 2020/5/10.
//

#ifndef CUDA_TEST_SCENEPARSER_CUH
#define CUDA_TEST_SCENEPARSER_CUH
#ifndef SCENE_PARSER_H
#define SCENE_PARSER_H

#include <cassert>
#include <fstream>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

#include "vec/vec_mat.cuh"
#include "camera.cuh"
#include "material.cuh"
#include "object3d.cuh"
#include "objects.cuh"
#include "mesh.cuh"
#include "sphere.cuh"
#include "plane.cuh"
#include "triangle.cuh"
#include "transform.cuh"
//#include "curve.cuh"
//#include "revsurface.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DegreesToRadians(x) ((M_PI * x) / 180.0f)
class Curve;
class RevSurface;

#define MAX_PARSER_TOKEN_LENGTH 1024



class SceneParser {
public:

    SceneParser() = delete;
    SceneParser(const char *filename) {
        // initialize some reasonable default values
        group = nullptr;
        camera = nullptr;
        current_material = nullptr;

        // parse the file
        assert(filename != nullptr);
        const char *ext = &filename[strlen(filename) - 4];

        if (strcmp(ext, ".txt") != 0) {
            printf("wrong file name extension\n");
            exit(0);
        }
        file = fopen(filename, "r");

        if (file == nullptr) {
            printf("cannot open scene file\n");
            exit(0);
        }
        parseFile();
        fclose(file);
        file = nullptr;
    }

    ~SceneParser() {
        delete group;
        delete camera;

        for(auto& m : materials) {
            delete m;
        }
        for(auto& k :kdtrees) {
            delete k;
        }
    }

    Camera *getCamera() const {
        return camera;
    }



    Material *getMaterial(int i) const {
        assert(i >= 0 && i < materials.size());
        return materials[i];
    }

    Objects *getGroup() const {
        return group;
    }

private:

    void parseFile() {
        char token[MAX_PARSER_TOKEN_LENGTH];
        while (getToken(token)) {
            if (!strcmp(token, "PerspectiveCamera")) {
                parsePerspectiveCamera();
            } else if (!strcmp(token, "Materials")) {
                parseMaterials();
            } else if (!strcmp(token, "Group")) {
                parseGroup();
            } else {
                printf("Unknown token in parseFile: '%s'\n", token);
                exit(0);
            }
        }
    }

    void parsePerspectiveCamera() {
        char token[MAX_PARSER_TOKEN_LENGTH];
        // read in the camera parameters
        getToken(token);
        assert (!strcmp(token, "{"));
        getToken(token);
        assert (!strcmp(token, "center"));
        Vector3f center = readVector3f();
        getToken(token);
        assert (!strcmp(token, "direction"));
        Vector3f direction = readVector3f();
        getToken(token);
        assert (!strcmp(token, "angle"));
        float angle_degrees = readFloat();
        float angle_radians = DegreesToRadians(angle_degrees);
        getToken(token);
        assert (!strcmp(token, "width"));
        int width = readInt();
        getToken(token);
        assert (!strcmp(token, "height"));
        int height = readInt();
        getToken(token);
        assert (!strcmp(token, "flength"));
        int flength = readFloat();
        getToken(token);
        assert (!strcmp(token, "aperture"));
        int aperture = readFloat();
        getToken(token);
        assert (!strcmp(token, "}"));
        camera = new Camera(center, direction, width, height, angle_radians, flength, aperture);
    }

    void parseMaterials() {
        char token[MAX_PARSER_TOKEN_LENGTH];
        getToken(token);
        assert (!strcmp(token, "{"));
        // read in the number of objects
        getToken(token);
        assert (!strcmp(token, "numMaterials"));
        int num_materials = readInt();
        materials.resize(num_materials, nullptr);
        // read in the objects
        int count = 0;
        while (num_materials > count) {
            getToken(token);
            if (!strcmp(token, "Material")) {
                materials[count] = parseMaterial();
            } else {
                printf("Unknown token in parseMaterial: '%s'\n", token);
                exit(0);
            }
            count++;
        }
        getToken(token);
        assert (!strcmp(token, "}"));
    }
    Material *parseMaterial() {
        char token[MAX_PARSER_TOKEN_LENGTH];
        char filename[MAX_PARSER_TOKEN_LENGTH];
        filename[0] = 0;
        Vector3f emission = make_float3(0,0,0), color = make_float3(1,1,1);
        REFL_TYPE reflType = DIFF;
        Texture* texture = nullptr;
        getToken(token);
        assert (!strcmp(token, "{"));
        while (true) {
            getToken(token);
            if (strcmp(token, "emission") == 0) {
                emission = readVector3f();
            } else if (strcmp(token, "color") == 0) {
                color = readVector3f();
            } else if (strcmp(token, "type") == 0) {
                getToken(token);
                if (strcmp(token, "diff") == 0) {
                    reflType = DIFF;
                } else if (strcmp(token, "refr") == 0) {
                    reflType = REFR;
                } else if (strcmp(token, "spec") == 0) {
                    reflType = SPEC;
                } else {
                    printf("Unknown token in parseMaterial: '%s'\n", token);
                    exit(0);
                }
            } else if (strcmp(token, "texture") == 0) {
                // TODO: Optional: read in texture and draw it.
                //char cf[MAX_PARSER_TOKEN_LENGTH], ef[MAX_PARSER_TOKEN_LENGTH], tf[MAX_PARSER_TOKEN_LENGTH], bf[MAX_PARSER_TOKEN_LENGTH];
                char* cf = nullptr, *ef = nullptr, *tf = nullptr, *bf = nullptr;
                getToken(token);
                assert (!strcmp(token, "{"));
                while (true) {
                    getToken(token);
                    if (strcmp(token, "color") == 0) {
                        cf = new char[MAX_PARSER_TOKEN_LENGTH];
                        getToken(cf);
                    }
                    else if (strcmp(token, "emission") == 0) {
                        ef = new char[MAX_PARSER_TOKEN_LENGTH];
                        getToken(ef);
                    }
                    else if (strcmp(token, "type") == 0) {
                        tf = new char[MAX_PARSER_TOKEN_LENGTH];
                        getToken(tf);
                    }
                    else if (strcmp(token, "bump") == 0) {
                        bf = new char[MAX_PARSER_TOKEN_LENGTH];
                        getToken(bf);
                    }
                    else {
                        assert(!strcmp(token, "}"));
                        break;
                    }
                }
                texture = new Texture(ef, cf, tf, bf);
                delete cf,ef,tf,bf;
            } else {
                assert (!strcmp(token, "}"));
                break;
            }
        }
        auto *answer = new Material(emission, color, reflType, texture);

        return answer;
    }

    void parseGroup() {
        char token[MAX_PARSER_TOKEN_LENGTH];
        getToken(token);
        assert (!strcmp(token, "{"));

        // read in the number of objects
        getToken(token);
        assert (!strcmp(token, "numObjects"));
        int num_objects = readInt();

        this->group = new Objects(num_objects);

        // read in the objects
        int count = 0;
        while (num_objects > count) {
            getToken(token);
            if (!strcmp(token, "MaterialIndex")) {
                // change the current material
                int index = readInt();
                assert (index >= 0 && index <= materials.size());
                current_material = getMaterial(index);
            } else {
                // TODO: parse right here
                if (!strcmp(token, "Sphere")) {
                    assert(parseSphere(&this->group));
                } else if (!strcmp(token, "Plane")) {
                    assert(parsePlane(&this->group));
                } else if (!strcmp(token, "Mesh")) {
                    assert(parseTriangleMesh(&this->group));
                } else if (!strcmp(token, "Transform")) {
                    assert(parseTransformedMesh(&this->group));
                //} //else if (!strcmp(token, "RevSurface")) {
//                    assert(parseRevSurface(&this->group));
                } else {
                    printf("Unknown token in parseGroup: '%s'\n", token);
                    exit(0);
                }
                count++;
            }
        }
        getToken(token);
        assert (!strcmp(token, "}"));
        this->group->initBvh();
    }
    bool parseSphere(Objects** objects) {
        char token[MAX_PARSER_TOKEN_LENGTH];
        getToken(token);
        assert (!strcmp(token, "{"));
        getToken(token);
        assert (!strcmp(token, "center"));
        Vector3f center = readVector3f();
        getToken(token);
        assert (!strcmp(token, "radius"));
        float radius = readFloat();
        getToken(token);
        assert (!strcmp(token, "}"));
        assert (current_material != nullptr);
        (*objects)->addSphere(center, radius, current_material);
        return true;
    }
    bool parsePlane(Objects** objects) {
        char token[MAX_PARSER_TOKEN_LENGTH];
        getToken(token);
        assert (!strcmp(token, "{"));
        getToken(token);
        assert (!strcmp(token, "normal"));
        Vector3f normal = readVector3f();
        getToken(token);
        assert (!strcmp(token, "offset"));
        float offset = readFloat();
        getToken(token);
        assert (!strcmp(token, "range"));
        int range = readInt();

        Vector3f centerp = make_float3(0,0,0), dirU = make_float3(0,0,0), dirV = make_float3(0,0,0);
        while (true) {
            getToken(token);
            if (strcmp(token, "centerp") == 0) {
                centerp = readVector3f();
            }
            else if (strcmp(token, "dirV") == 0) {
                dirV = readVector3f();
            }
            else if (strcmp(token, "dirU") == 0) {
                dirU = readVector3f();
            }
            else {
                assert(!strcmp(token, "}"));
                break;
            }
        }


//        getToken(token);
//        assert (!strcmp(token, "}"));
        assert (current_material != nullptr);
        (*objects)->addPlane(normal, offset, range, centerp, dirU, dirV, current_material);
        return true;
    }

    bool parseTriangleMesh(Objects** objects) {
        char token[MAX_PARSER_TOKEN_LENGTH];
        char filename[MAX_PARSER_TOKEN_LENGTH];
        // get the filename
        getToken(token);
        assert (!strcmp(token, "{"));
        getToken(token);
        assert (!strcmp(token, "obj_file"));
        getToken(filename);
        getToken(token);
        assert (!strcmp(token, "}"));
        const char *ext = &filename[strlen(filename) - 4];
        assert(!strcmp(ext, ".obj"));
        auto* kdTree = new KdTree(filename);
        kdtrees.push_back(kdTree);
        (*objects)->addMesh(kdTree, current_material);
        return true;
    }

    bool parseTransformedMesh(Objects** objects) {
        char token[MAX_PARSER_TOKEN_LENGTH];
        Matrix4f matrix = Matrix4f::identity();

        getToken(token);
        assert (!strcmp(token, "{"));
        // read in transformations:
        // apply to the LEFT side of the current matrix (so the first
        // transform in the list is the last applied to the object)
        getToken(token);


        while (true) {
            if (!strcmp(token, "Scale")) {
                Vector3f s = readVector3f();
                matrix = matrix * Matrix4f::scaling(s.x, s.y, s.z);
            } else if (!strcmp(token, "UniformScale")) {
                float s = readFloat();
                matrix = matrix * Matrix4f::uniformScaling(s);
            } else if (!strcmp(token, "Translate")) {
                Vector3f trans = readVector3f();
                matrix = matrix * Matrix4f::translation(trans.x, trans.y, trans.z);
            } else if (!strcmp(token, "XRotate")) {
                matrix = matrix * Matrix4f::rotateX(DegreesToRadians(readFloat()));
            } else if (!strcmp(token, "YRotate")) {
                matrix = matrix * Matrix4f::rotateY(DegreesToRadians(readFloat()));
            } else if (!strcmp(token, "ZRotate")) {
                matrix = matrix * Matrix4f::rotateZ(DegreesToRadians(readFloat()));
            } else if (!strcmp(token, "Rotate")) {
                getToken(token);
                assert (!strcmp(token, "{"));
                Vector3f axis = readVector3f();
                float degrees = readFloat();
                float radians = DegreesToRadians(degrees);
                matrix = matrix * Matrix4f::rotation(axis, radians);
                getToken(token);
                assert (!strcmp(token, "}"));
            } else if (!strcmp(token, "Matrix4f")) {
                Matrix4f matrix2 = Matrix4f::identity();
                getToken(token);
                assert (!strcmp(token, "{"));
                for (int j = 0; j < 4; j++) {
                    for (int i = 0; i < 4; i++) {
                        float v = readFloat();
                        matrix2(i, j) = v;
                    }
                }
                getToken(token);
                assert (!strcmp(token, "}"));
                matrix = matrix2 * matrix;
            } else if (!strcmp(token, "Mesh")) {

                char filename[MAX_PARSER_TOKEN_LENGTH];
                getToken(token);
                assert (!strcmp(token, "{"));
                getToken(token);
                assert (!strcmp(token, "obj_file"));
                getToken(filename);
                getToken(token);
                assert (!strcmp(token, "}"));
                const char *ext = &filename[strlen(filename) - 4];
                assert(!strcmp(ext, ".obj"));
                auto* kdTree = new KdTree(filename);
                kdtrees.push_back(kdTree);
                (*objects)->addTransformedMesh(kdTree, current_material, matrix);
                break;

            } else if (!strcmp(token, "RevSurface")) {
                getToken(token);
                assert (!strcmp(token, "{"));
                getToken(token);
                assert (!strcmp(token, "profile"));
                getToken(token);
                if (!strcmp(token, "BsplineCurve")) {
                    getToken(token);
                    assert (!strcmp(token, "{"));
                    getToken(token);
                    assert (!strcmp(token, "controls"));
                    vector<Vector3f> controls;
                    while (true) {
                        getToken(token);
                        if (!strcmp(token, "[")) {
                            controls.push_back(readVector3f());
                            getToken(token);
                            assert (!strcmp(token, "]"));
                        } else if (!strcmp(token, "}")) {
                            break;
                        } else {
                            printf("Incorrect format for BsplineCurve!\n");
                            exit(0);
                        }
                    }
                    Vector3f* controls_d;
                    cudaMallocManaged(&controls_d, sizeof(Vector3f) * controls.size());
                    for (int i = 0; i < controls.size(); i++) {
                        controls_d[i] = controls[i];
                    }
                    (*objects)->addTransformedRevsurface(controls_d, controls.size(), 3, current_material, matrix, 0);
                    cudaFree(controls_d);
                }
                else if (!strcmp(token, "BezierCurve")) {
                    getToken(token);
                    assert (!strcmp(token, "{"));
                    getToken(token);
                    assert (!strcmp(token, "controls"));
                    vector<Vector3f> controls;
                    while (true) {
                        getToken(token);
                        if (!strcmp(token, "[")) {
                            controls.push_back(readVector3f());
                            getToken(token);
                            assert (!strcmp(token, "]"));
                        } else if (!strcmp(token, "}")) {
                            break;
                        } else {
                            printf("Incorrect format for BezierCurve!\n");
                            exit(0);
                        }
                    }
                    Vector3f* controls_d;
                    cudaMallocManaged(&controls_d, sizeof(Vector3f) * controls.size());
                    for (int i = 0; i < controls.size(); i++) {
                        controls_d[i] = controls[i];
                    }
                    (*objects)->addTransformedRevsurface(controls_d, controls.size(), 3, current_material, matrix, 1);
                    cudaFree(controls_d);
                }
                else {
                    printf("Unknown profile type in parseRevSurface: '%s'\n", token);
                    exit(0);
                }
                getToken(token);
                assert (!strcmp(token, "}"));

                break;
            }

            getToken(token);
        }


        getToken(token);
        assert (!strcmp(token, "}"));
        return true;
    }

    bool parseBsplineCurve() {
        char token[MAX_PARSER_TOKEN_LENGTH];
        getToken(token);
        assert (!strcmp(token, "{"));
        getToken(token);
        assert (!strcmp(token, "controls"));
        vector<Vector3f> controls;
        while (true) {
            getToken(token);
            if (!strcmp(token, "[")) {
                controls.push_back(readVector3f());
                getToken(token);
                assert (!strcmp(token, "]"));
            } else if (!strcmp(token, "}")) {
                break;
            } else {
                printf("Incorrect format for BsplineCurve!\n");
                exit(0);
            }
        }

        Vector3f* controls_d;
        cudaMallocManaged(&controls_d, sizeof(Vector3f) * controls.size());
        for (int i = 0; i < controls.size(); i++) {
            controls_d[i] = controls[i];
        }
        //(*objects)->addTransformedRevsurface(controls_d, )
        cudaFree(controls_d);
        return true;
    }
//    bool parseRevSurface(Objects** objects) {
//        char token[MAX_PARSER_TOKEN_LENGTH];
//
//    }

    int getToken(char token[MAX_PARSER_TOKEN_LENGTH]) {
        // for simplicity, tokens must be separated by whitespace
        assert (file != nullptr);
        int success = fscanf(file, "%s ", token);
        if (success == EOF) {
            token[0] = '\0';
            return 0;
        }
        return 1;
    }

    Vector3f readVector3f() {
        float x, y, z;
        int count = fscanf(file, "%f %f %f", &x, &y, &z);
        if (count != 3) {
            printf("Error trying to read 3 floats to make a Vector3f\n");
            assert (0);
        }
        return make_float3(x, y, z);
    }

    float readFloat() {
        float answer;
        int count = fscanf(file, "%f", &answer);
        if (count != 1) {
            printf("Error trying to read 1 float\n");
            assert (0);
        }
        return answer;
    }
    int readInt() {
        int answer;
        int count = fscanf(file, "%d", &answer);
        if (count != 1) {
            printf("Error trying to read 1 int\n");
            assert (0);
        }
        return answer;
    }

    FILE *file;
    Camera *camera;

    vector<Material*> materials;
    vector<KdTree*> kdtrees;
    Material *current_material;
    Objects *group;
};

#endif // SCENE_PARSER_H

#endif //CUDA_TEST_SCENEPARSER_CUH

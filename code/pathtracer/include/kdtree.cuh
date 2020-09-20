//
// Created by blackcloud on 2020/5/7.
//

#ifndef CUDA_TEST_KDTREE_CUH
#define CUDA_TEST_KDTREE_CUH
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <utility>
#include <sstream>

#include "ray.cuh"
#include "aabb.cuh"
#include "triangle.cuh"
#include "managed.cuh"

// max of 2^32 * 10 triangles
#define MAX_KDTREE_DEPTH 15
#define MIN_TRIANGLE_NUM_LIMIT 15

using std::map;
using std::vector;

class KdNode : public Managed {
public:
    KdNode() = delete ;
    KdNode(Triangle* triangles, const vector<int>& triangle_idx, const AABB& aabbParent, const int level, const int dimension) {

        // triangle_idx & size
        this->triangle_idx_size = triangle_idx.size();
        this->triangle_idx = nullptr;

        // AABB
        this->aabb = aabbParent;
        //level
        this->level = level;
        //dim
        this->dimension = dimension;

        // create next node or not
        if (level < MAX_KDTREE_DEPTH && triangle_idx.size() > MIN_TRIANGLE_NUM_LIMIT) {
            double splitPoint = generateSplit(triangles, triangle_idx);

            vector<int> leftindices;
            vector<int> rightindices;

            AABB aabbLeft;
            AABB aabbRight;
            if (dimension == 0) {
                aabbLeft = AABB(aabb.getMin(), make_float3(splitPoint, aabb.getMax().y, aabb.getMax().z));
                aabbRight = AABB(make_float3(splitPoint, aabb.getMin().y, aabb.getMin().z), aabb.getMax());
            } else if (dimension == 1) {
                aabbLeft = AABB(aabb.getMin(), make_float3(aabb.getMax().x, splitPoint, aabb.getMax().z));
                aabbRight = AABB(make_float3(aabb.getMin().x, splitPoint, aabb.getMin().z), aabb.getMax());
            } else {
                aabbLeft = AABB(aabb.getMin(), make_float3(aabb.getMax().x, aabb.getMax().y, splitPoint));
                aabbRight = AABB(make_float3(aabb.getMin().x, aabb.getMin().y, splitPoint), aabb.getMax());
            }

            for (const int index : triangle_idx) {
                if (aabbLeft.isIntersect(triangles[index])) {
                    leftindices.push_back(index);
                }
                if (aabbRight.isIntersect(triangles[index])) {
                    rightindices.push_back(index);
                }
            }
            int next_dimension = (dimension + 1) % 3;

            left = new KdNode(triangles, leftindices, aabbLeft, level + 1, next_dimension);
            right = new KdNode(triangles, rightindices, aabbRight, level + 1, next_dimension);
        } else {
            leaf = true;
            cudaMallocManaged(&this->triangle_idx, sizeof(int) * triangle_idx_size);
            for (int i = 0; i < triangle_idx_size; i++) {
                this->triangle_idx[i] = triangle_idx[i];
            }
        }
    }

    KdNode(const KdNode& node) {
        // triangle_idx & size
        this->triangle_idx_size = node.triangle_idx_size;


        // AABB
        this->aabb = node.aabb;
        //level
        this->level = node.level;
        //dim
        this->dimension = node.dimension;
        //leaf
        this->leaf = node.leaf;
        if (this->leaf) {
            cudaMallocManaged(&this->triangle_idx, sizeof(int) * triangle_idx_size);
            for (int i = 0; i < triangle_idx_size; i++) {
                this->triangle_idx[i] = node.triangle_idx[i];
            }
        } else {
            this->triangle_idx = nullptr;
        }
        this->left = new KdNode(*node.left);
        this->right = new KdNode(*node.right);
    }

    ~KdNode() {
        if (this->triangle_idx)
            cudaFree(this->triangle_idx);
        delete this->left;
        delete this->right;
    }

    __host__ __device__ const AABB& getBox() const {
        return aabb;
    }
    __host__ __device__ KdNode* getLeftChild() const {
        return left;
    }
    __host__ __device__ KdNode* getRightChild() const {
        return right;
    }
    __host__ __device__ inline bool isLeaf() const {
        return leaf;
    }
    __host__ __device__ inline int getTrianglesSize() const {
        return triangle_idx_size;
    }
    __host__ __device__ inline int* getTriangleIndex() const {
        return triangle_idx;
    }
private:
    double generateSplit(Triangle* triangles, const vector<int>& curr_triangles) const {
        map<double, int> map;

        for (int index : curr_triangles) {
            map[getDimension(triangles[index].getCentroid(), dimension)] = index;
        }

        auto iterator = map.begin();
        advance(iterator, map.size() / 2);
        return getDimension(triangles[iterator->second].getCentroid(), dimension);
    }


    int level;
    int dimension;
    KdNode *left = nullptr;
    KdNode *right = nullptr;

    int *triangle_idx;
    int triangle_idx_size;

    AABB aabb;
    bool leaf = false;
};

void loadObj(const char* filename, vector<Triangle>& faces);

class KdTree : public Managed {
public:
    KdTree() = delete ;
    explicit KdTree(const char* filename) {
        //printf("this %d\n", triangles_size);
        vector<Triangle> faces;
        loadObj(filename, faces);

        this->triangles_size = faces.size();
        cudaMallocManaged(&this->triangles, sizeof(Triangle) * this->triangles_size);

        double minX, maxX, minY, maxY, minZ, maxZ;
        minX = triangles[0].vertices[0].x;
        maxX = triangles[0].vertices[0].x;
        minY = triangles[0].vertices[0].y;
        maxY = triangles[0].vertices[0].y;
        minZ = triangles[0].vertices[0].z;
        maxZ = triangles[0].vertices[0].z;
        vector<int> trianglePointers(triangles_size);
        // init point min_point and max_point, for the sake of creating AABB box

        for(int i = 0; i < this->triangles_size; i++) {
            const Triangle& triangle = faces[i];
            this->triangles[i] = triangle;
            trianglePointers[i] = i;

            maxX = fmax(triangle.vertices[0].x, maxX);
            minX = fmin(triangle.vertices[0].x, minX);
            maxX = fmax(triangle.vertices[1].x, maxX);
            minX = fmin(triangle.vertices[1].x, minX);
            maxX = fmax(triangle.vertices[2].x, maxX);
            minX = fmin(triangle.vertices[2].x, minX);

            maxY = fmax(triangle.vertices[0].y, maxY);
            minY = fmin(triangle.vertices[0].y, minY);
            maxY = fmax(triangle.vertices[1].y, maxY);
            minY = fmin(triangle.vertices[1].y, minY);
            maxY = fmax(triangle.vertices[2].y, maxY);
            minY = fmin(triangle.vertices[2].y, minY);

            maxZ = fmax(triangle.vertices[0].z, maxZ);
            minZ = fmin(triangle.vertices[0].z, minZ);
            maxZ = fmax(triangle.vertices[1].z, maxZ);
            minZ = fmin(triangle.vertices[1].z, minZ);
            maxZ = fmax(triangle.vertices[2].z, maxZ);
            minZ = fmin(triangle.vertices[2].z, minZ);
        }
        // biggest AABB box as root
        mainNode = new KdNode(triangles, trianglePointers, AABB(make_float3(minX, minY, minZ), make_float3(maxX, maxY, maxZ)), 0, 0);
    }

    KdTree(const KdTree& kdtree) {
        this->triangles_size = kdtree.triangles_size;
        cudaMallocManaged(&triangles, sizeof(Triangle) * this->triangles_size);
        for (int i = 0; i < triangles_size; i++) {
            this->triangles[i] = kdtree.triangles[i];
        }
        mainNode = new KdNode(*mainNode);
    }

    ~KdTree() {
        delete mainNode;
        cudaFree(triangles);
    }


    __device__ void intersect(const Ray& r, Hit& h, float tmin, BoolBitField* mark, const int mark_size, KdNode* node, bool& result) const {
        const int idx_size = node->getTrianglesSize();
        const int* idx = node->getTriangleIndex();
        for (int i = 0; i < idx_size; i++) {
            int index = idx[i];
            int _x = index / 8;
            int _y = index % 8;
            //if (true) {
            if (_x >= mark_size || !mark[_x][_y]) {
                //printf("Index: %d, Norm ",index)
                result |= triangles[index].intersect(r, h, tmin);
                if (_x < mark_size) {
                    switch (_y) {
                        case 0:
                            mark[_x].b0 = true;
                            break;
                        case 1:
                            mark[_x].b1 = true;
                            break;
                        case 2:
                            mark[_x].b2 = true;
                            break;
                        case 3:
                            mark[_x].b3 = true;
                            break;
                        case 4:
                            mark[_x].b4 = true;
                            break;
                        case 5:
                            mark[_x].b5 = true;
                            break;
                        case 6:
                            mark[_x].b6 = true;
                            break;
                        case 7:
                            mark[_x].b7 = true;
                            break;
                        default:
                            break;
                    }
                }
            }
        }
    }
    __device__ bool intersect(const Ray& r, Hit& h, float tmin, ThreadResource* thread_resource) const {
        BoolBitField *mark = thread_resource->_repeat_mark; // pre allocated
        int mark_size = thread_resource->_mark_size;
        memset(mark, 0, sizeof(BoolBitField) * mark_size);

        bool result = false;
        KdNode* stack[MAX_KDTREE_DEPTH * 2];
        KdNode** stackPtr = stack;
        *(stackPtr++) = nullptr;

        if (!mainNode->getLeftChild() && !mainNode->getRightChild()) {
            intersect(r, h, tmin, mark, mark_size, mainNode, result);
        }

        KdNode* node = mainNode;
        do {
            KdNode* lchild = node->getLeftChild();
            KdNode* rchild = node->getRightChild();
            bool lcNull = (lchild == nullptr);
            bool rcNull = (rchild == nullptr);

            bool intersectL = lcNull ? false : lchild->getBox().isIntersect(r);
            bool intersectR = rcNull ? false : rchild->getBox().isIntersect(r);
//            bool intersectL = !lcNull;
//            bool intersectR = !rcNull;

            // add idx
            if (intersectL && lchild->isLeaf()) {
                intersect(r, h, tmin, mark, mark_size, lchild, result);
            }

            if (intersectR && rchild->isLeaf()) {
                intersect(r, h, tmin, mark, mark_size, rchild, result);
            }

            bool traverseL = (intersectL && !lchild->isLeaf());
            bool traverseR = (intersectR && !rchild->isLeaf());

            if (!traverseL && !traverseR) {
                node = *(--stackPtr); //pop
            }
            else {
                node = (traverseL) ? lchild : rchild;
                if (traverseL && traverseR) {
                    *(stackPtr++) = rchild; //push
                }
            }
        }
        while (node != nullptr);
        return result;
    }

    __device__ KdNode* getRoot() const {
        return mainNode;
    }
    int getTriangleSize() const {
        return triangles_size;
    }
private:
    Triangle *triangles;
    int triangles_size;
    KdNode *mainNode;
};





struct TriangleIndex {
    TriangleIndex() {
        x[0] = 0; x[1] = 0; x[2] = 0;
    }
    int &operator[](const int i) { return x[i]; }
    int x[3]{};
};




void loadObj(const char* filename, vector<Triangle>& faces) {
    faces.clear();
    vector<Vector3f> vertexes;
    std::ifstream f;
    f.open(filename);
    if (!f.is_open()) {
        std::cout << "Cannot open " << filename << "\n";
        return;
    }
    std::string line;
    std::string vTok("v");
    std::string fTok("f");
    std::string texTok("vt");
    char bslash = '/', space = ' ';
    std::string tok;
    int texID;
    while (true) {
        std::getline(f, line);
        if (f.eof()) {
            break;
        }
        if (line.size() < 3) {
            continue;
        }
        if (line.at(0) == '#') {
            continue;
        }
        std::stringstream ss(line);
        ss >> tok;
        if (tok == vTok) {
            float x,y,z;
            //Vector3f vec;
            ss >> x >> y >> z;
            vertexes.push_back(make_float3(x,y,z));
        }
        else if (tok == fTok) {
            if (line.find(bslash) != std::string::npos) {
                std::replace(line.begin(), line.end(), bslash, space);
                std::stringstream facess(line);
                TriangleIndex trig;
                facess >> tok;
                for (int ii = 0; ii < 3; ii++) {
                    facess >> trig[ii] >> texID;
                    trig[ii]--;
                }
                faces.emplace_back(Triangle(vertexes[trig[0]], vertexes[trig[1]], vertexes[trig[2]]));
            } else {
                TriangleIndex trig;
                for (int ii = 0; ii < 3; ii++) {
                    ss >> trig[ii];
                    trig[ii]--;
                }
                faces.emplace_back(Triangle(vertexes[trig[0]], vertexes[trig[1]], vertexes[trig[2]]));
            }
        }
        else if (tok == texTok) {
            //Vector2f texcoord;
            float _texcoord;
            ss >> _texcoord;
            ss >> _texcoord;
        }
    }
    f.close();
}
#endif //CUDA_TEST_KDTREE_CUH

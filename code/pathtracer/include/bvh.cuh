//
// Created by blackcloud on 2020/5/9.
//

#ifndef CUDA_TEST_BVH_CUH
#define CUDA_TEST_BVH_CUH
#include "managed.cuh"
#include "object3d.cuh"
#include <cmath>
#include <vector>
#include <algorithm>
#define MAX_BVH_TREE_DEPTH 20
using std::vector;
using std::pair;
// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}


// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
unsigned int morton3D(const float3& point)
{
    float x = point.x, y = point.y, z = point.z;
    x = fmin(fmax(x * 1024.0f, 0.0f), 1023.0f);
    y = fmin(fmax(y * 1024.0f, 0.0f), 1023.0f);
    z = fmin(fmax(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}


class BvhNode : public Managed {
private:
    bool leaf = false;
    BvhNode* left = nullptr, *right = nullptr;
    int obj_id = -1;
    AABB aabb;


    static int findSplit( const vector<unsigned int>& sortedMortonCodes, const int first, const int last) {
        // Identical Morton codes => split the range in the middle.

        unsigned int firstCode = sortedMortonCodes[first];
        unsigned int lastCode = sortedMortonCodes[last];

        if (firstCode == lastCode)
            return (first + last) >> 1;

        // Calculate the number of highest bits that are the same
        // for all objects, using the count-leading-zeros intrinsic.

        int commonPrefix = __builtin_clz(firstCode ^ lastCode);

        // Use binary search to find where the next bit differs.
        // Specifically, we are looking for the highest object that
        // shares more than commonPrefix bits with the first one.

        int split = first; // initial guess
        int step = last - first;

        do
        {
            step = (step + 1) >> 1; // exponential decrease
            int newSplit = split + step; // proposed new position

            if (newSplit < last)
            {
                unsigned int splitCode = sortedMortonCodes[newSplit];
                int splitPrefix = __builtin_clz(firstCode ^ splitCode);
                if (splitPrefix > commonPrefix)
                    split = newSplit; // accept proposal
            }
        }
        while (step > 1);
        return split;
    }

public:
    BvhNode(const AABB* aabb_of_objs,
            const vector<unsigned int>& sortedMortonCodes,
            const vector<int>&          sortedObjectIDs,
            const int           first,
            const int           last) {
//        printf("BvhNode\n");
        //this->aabb = myaabb;

        // calc aabb
//        printf("First: %d\n",first);
//        printf("Last: %d\n",last);
//        printf("First Id: %d\n", sortedObjectIDs[first]);

        const AABB first_box = aabb_of_objs[sortedObjectIDs[first]];

        double minX, maxX, minY, maxY, minZ, maxZ;
        minX = first_box.getMin().x;
        maxX = first_box.getMax().x;
        minY = first_box.getMin().y;
        maxY = first_box.getMax().y;
        minZ = first_box.getMin().z;
        maxZ = first_box.getMax().z;

//        printf("Init min:%f,%f,%f\n", minX,minY,minZ);
//        printf("Init max:%f,%f,%f\n", maxX,maxY,maxZ);

        // init point min_point and max_point, for the sake of creating AABB box
        for(int i = first; i <= last; i++) {
            //const Object3D* obj = objects[sortedObjectIDs[i]];
            const AABB box = aabb_of_objs[sortedObjectIDs[i]];
            const Vector3f min_point = box.getMin();
            const Vector3f max_point = box.getMax();
            maxX = fmax(max_point.x, maxX);
            minX = fmin(min_point.x, minX);
            maxY = fmax(max_point.y, maxY);
            minY = fmin(min_point.y, minY);
            maxZ = fmax(max_point.z, maxZ);
            minZ = fmin(min_point.z, minZ);
        }

        this->aabb = AABB(make_float3(minX,minY,minZ), make_float3(maxX,maxY,maxZ));
//        printf("Final min:%f,%f,%f\n", minX,minY,minZ);
//        printf("Final max:%f,%f,%f\n", maxX,maxY,maxZ);
        if (first == last) {
            leaf = true;
            obj_id = sortedObjectIDs[first];
            return;
        }

        // Determine where to split the range.
        int split = findSplit(sortedMortonCodes, first, last);

        // Process the resulting sub-ranges recursively.
        left = new BvhNode(aabb_of_objs, sortedMortonCodes, sortedObjectIDs, first, split);
        right = new BvhNode(aabb_of_objs, sortedMortonCodes, sortedObjectIDs, split + 1, last);
    }
    BvhNode(const BvhNode& bvhNode) {
        //printf("Copy node called\n");
        aabb = bvhNode.aabb;
        leaf = bvhNode.leaf;
        obj_id = bvhNode.obj_id;
        left = new BvhNode(*bvhNode.left);
        right = new BvhNode(*bvhNode.right);
    }
    ~BvhNode(){
        delete this->left;
        delete this->right;
    }
    __device__ inline bool isLeaf() const  {
        return leaf;
    }
    __host__ __device__ const AABB& getBox() const {
        return aabb;
    }
    __host__ __device__ BvhNode* getLeftChild() const {
        return left;
    }
    __host__ __device__ BvhNode* getRightChild() const {
        return right;
    }
    __device__ int getIndex() const {
        return obj_id;
    }

};

class BvhTree : public Managed {
private:
    Object3D** objects;
    int objects_num;
    BvhNode* root;

public:
    BvhNode* getRoot() const {
        return root;
    }
    BvhTree(Object3D** objs, int objs_num, const AABB* aabb_of_objs) : objects(objs), objects_num(objs_num) {
        //printf("BvhTree\n");
        assert(1<<MAX_BVH_TREE_DEPTH > objs_num);

        // calc morton codes
        vector<pair<int, unsigned int> > Id_mortonCodes;
        for (int i = 0; i < objs_num; i++) {
            int id = i;
            int mortoncode = morton3D(aabb_of_objs[i].getCenter());
            Id_mortonCodes.emplace_back(id, mortoncode);
        }
        auto cmp = [&](const pair<int, unsigned int>& p1, const pair<int, unsigned int>& p2){
            return p1.second < p2.second;
        };
        std::sort(Id_mortonCodes.begin(), Id_mortonCodes.end(), cmp);
        vector<unsigned int> sortedmortoncodes;
        vector<int> sortedidx;

        for (auto & Id_mortonCode : Id_mortonCodes) {
            sortedmortoncodes.push_back(Id_mortonCode.second);
            sortedidx.push_back(Id_mortonCode.first);
        }

        for (int i = 0; i < objs_num; i++) {
            auto min = aabb_of_objs[i].getMin();
            auto max = aabb_of_objs[i].getMax();
//            printf("Id: %d\n",i);
//            printf("Min %f,%f,%f\n", min.x, min.y, min.z);
//            printf("Max %f,%f,%f\n", max.x, max.y, max.z);
        }

        root = new BvhNode(aabb_of_objs, sortedmortoncodes, sortedidx, 0, Id_mortonCodes.size()-1);
    }
    BvhTree(const BvhTree& bvhTree) {
        objects = bvhTree.objects;
        objects_num = bvhTree.objects_num;
        root = new BvhNode(*bvhTree.root);
    }
    ~BvhTree() {
        delete root;
    }

//    __device__ void debug_log(const Ray& r, Hit& h, int& debug_cnt, int index) const {
////        printf("Count: %d   ID: %d\n", debug_cnt++, index);
////        printf("Ray: %f, %f, %f -> %f, %f, %f\n", r.getOrigin().x, r.getOrigin().y, r.getOrigin().z, r.getDirection().x, r.getDirection().y, r.getDirection().z);
////        printf("Hit Norm: %f, %f, %f\n", h.getNormal().x, h.getNormal().y, h.getNormal().z);
////        printf("Hit t: %f\n\n", h.getT());
//    }
    __device__ bool intersect(const Ray& r, Hit& h, float tmin, ThreadResource* resource) const {
        //printf("Intersect\n");
        bool result = false;
        BvhNode* stack[MAX_BVH_TREE_DEPTH * 2];
        BvhNode** stackPtr = stack;
        *(stackPtr++) = nullptr;

        //int debug_cnt = 0;

        if (root->isLeaf()) {
            result |= objects[root->getIndex()]->intersect(r,h,tmin,resource);
        }


        BvhNode* node = root;
        do {
            BvhNode* lchild = node->getLeftChild();
            BvhNode* rchild = node->getRightChild();
            bool lcNull = (lchild == nullptr);
            bool rcNull = (rchild == nullptr);


            bool intersectL = lcNull ? false : lchild->getBox().isIntersect(r);
            bool intersectR = rcNull ? false : rchild->getBox().isIntersect(r);


            if (intersectL && lchild->isLeaf()) {
                result |= objects[lchild->getIndex()]->intersect(r,h,tmin,resource);
            }

            if (intersectR && rchild->isLeaf()) {
                result |= objects[rchild->getIndex()]->intersect(r,h,tmin,resource);
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
};
#endif //CUDA_TEST_BVH_CUH

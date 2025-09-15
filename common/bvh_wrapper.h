#ifndef BVH_WRAPPER_H
#define BVH_WRAPPER_H


#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/node.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/tri.h>

#include <iostream>

#include "types.h"

namespace BVH_WRAPPER {


using Scalar  = double;
using Vec3    = bvh::v2::Vec<Scalar, 3>;
using BBox    = bvh::v2::BBox<Scalar, 3>;
using Tri     = bvh::v2::Tri<Scalar, 3>;
using Node    = bvh::v2::Node<Scalar, 3>;
using Bvh     = bvh::v2::Bvh<Node>;
using Ray     = bvh::v2::Ray<Scalar, 3>;

Vec3 toVec3(const vec& x){
    return Vec3(x(0),x(1),x(2));
}


inline auto& get(Tri& t,int i)
{
    if (i == 0)
        return t.p0;
    else if (i == 1)
        return t.p1;
    else
        return t.p2;
}

using PrecomputedTri = bvh::v2::PrecomputedTri<Scalar>;

class BVH{
public:
    BVH(){}
    BVH(const std::vector<Tri>& tris) {
        bvh::v2::ThreadPool thread_pool;
        bvh::v2::ParallelExecutor executor(thread_pool);

        // Get triangle centers and bounding boxes (required for BVH builder)
        std::vector<BBox> bboxes(tris.size());
        std::vector<Vec3> centers(tris.size());
        executor.for_each(0, tris.size(), [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                bboxes[i]  = tris[i].get_bbox();
                centers[i] = tris[i].get_center();
            }
        });

        typename bvh::v2::DefaultBuilder<Node>::Config config;
        config.quality = bvh::v2::DefaultBuilder<Node>::Quality::High;
        bvh = bvh::v2::DefaultBuilder<Node>::build(thread_pool, bboxes, centers, config);


        // This precomputes some data to speed up traversal further.
        precomputed_tris.resize(tris.size());
        executor.for_each(0, tris.size(), [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                auto j = should_permute ? bvh.prim_ids[i] : i;
                precomputed_tris[i] = tris[j];
            }
        });

    }
    struct intersection {
        size_t id;
        Scalar u,v,distance;
        bool valid = true;
    };
    intersection get_intersection(const Vec3& origin,const Vec3& dir){
        auto ray = Ray {
            origin, // Ray origin
            dir, // Ray direction
            0.,               // Minimum intersection distance
            100.              // Maximum intersection distance
        };

        static constexpr size_t invalid_id = std::numeric_limits<size_t>::max();
        static constexpr size_t stack_size = 64;
        static constexpr bool use_robust_traversal = false;

        auto prim_id = invalid_id;
        Scalar u, v;

        // Traverse the BVH and get the u, v coordinates of the closest intersection.
        bvh::v2::SmallStack<Bvh::Index, stack_size> stack;
        bvh.intersect<false, use_robust_traversal>(ray, bvh.get_root().index, stack,
                                                   [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                size_t j = should_permute ? i : bvh.prim_ids[i];
                if (auto hit = precomputed_tris[j].intersect(ray)) {
                    prim_id = j;
                    std::tie(u, v) = *hit;
                }
            }
            return prim_id != invalid_id;
        });

        if (prim_id != invalid_id) {
            return {prim_id,u,v,ray.tmax,true};
        } else {
            //std::cerr << "No intersection found" << std::endl;
            return {0,0,0,0,false};
        }

    }
private:
    // Permuting the primitive data allows to remove indirections during traversal, which makes it faster.
    static constexpr bool should_permute = false;
    std::vector<PrecomputedTri> precomputed_tris;
    Bvh bvh;

};


}

#endif // BVH_WRAPPER_H

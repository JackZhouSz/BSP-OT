#include "bvh_wrapper.h"

BVH_WRAPPER::BVH::BVH(const std::vector<Tri> &tris)
{
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

BVH_WRAPPER::BVH::intersection BVH_WRAPPER::BVH::get_intersection(const Vec3 &origin, const Vec3 &dir)
{
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
        return {prim_id,u,v,ray.tmax};
    } else {
        std::cerr << "No intersection found" << std::endl;
        return {};
    }

}

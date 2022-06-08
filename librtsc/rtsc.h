#ifndef RTSC_H
#define RTSC_H

#include <TriMesh.h>
#include <utility>
#include <vector>

struct isoline_params
{
    const std::vector<float> val;
    const std::vector<float> test_num;
    const std::vector<float> test_den;
    const std::vector<float> ndotv;
    bool                     do_bfcull{false};
    bool                     do_hermite{false};
    bool                     do_test{false};
    float                    fade{0.0f};
};

std::pair<std::vector<trimesh::point3>, std::vector<trimesh::vec4>>
compute_isolines(const isoline_params& params, trimesh::TriMesh* mesh,
                 const trimesh::point& viewpos, const trimesh::vec& currcolor);

#endif

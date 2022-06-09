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

std::vector<float> compute_ndotv(const trimesh::TriMesh& mesh,
                                 trimesh::point3         view_pos);
std::vector<float> compute_kr(const trimesh::TriMesh& mesh,
                              trimesh::point3         view_pos);
std::pair<std::vector<float>, std::vector<float>> compute_schtest(
    const trimesh::TriMesh& mesh, const std::vector<float>& sctest_den,
    trimesh::point3 view_pos, float scthresh, float shthresh,
    bool extra_sin2theta);
std::pair<std::vector<float>, std::vector<trimesh::vec2>> compute_q1t1(
    const trimesh::TriMesh& mesh, trimesh::point view_pos,
    const std::vector<float>& ndotv);

#endif

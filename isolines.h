#ifndef ISOLINES_H
#define ISOLINES_H

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

void draw_isolines(const isoline_params& params, trimesh::TriMesh* themesh,
                   const trimesh::point& viewpos,
                   const trimesh::vec&   currcolor);

#endif

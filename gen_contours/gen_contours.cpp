//
// Copyright 2022 Hypothetic Inc.
// All Rights Reserved
//
// Software to output vector strokes corresponding to important contours of a 3d
// model

#include <TriMesh_algo.h>
#include <cstdio>
#include <rtsc.h>
#include <string>

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        printf("Usage: %s model_file_name output_file_name\n"
               "Convert a 3D model to vector strokes\n",
               argv[0]);
        return -1;
    }

    std::string model_file_name{argv[1]};
    std::string output_file_name{argv[2]};

    printf("Model File: %s\n", model_file_name.c_str());
    printf("Output File: %s\n", output_file_name.c_str());

    trimesh::TriMesh* mesh = trimesh::TriMesh::read(model_file_name.c_str());
    if (mesh == nullptr)
    {
        printf("Mesh load failed\n");
        return -1;
    }
    mesh->need_bsphere();
    auto  bsphere = mesh->bsphere;
    float fov     = 1.0;
    auto  xf      = trimesh::xform::trans(0, 0, -3.5f / fov * mesh->bsphere.r) *
              trimesh::xform::trans(-mesh->bsphere.center);
    
    return 0;
}
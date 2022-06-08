#include <Vec.h>
#include <mathutil.h>
#include "rtsc.h"

// Compute gradient of (kr * sin^2 theta) at vertex i
static inline trimesh::vec gradkr(int i, trimesh::TriMesh* mesh,
                                  const trimesh::point& viewpos)
{
    trimesh::vec viewdir      = viewpos - mesh->vertices[i];
    float        rlen_viewdir = 1.0f / len(viewdir);
    viewdir *= rlen_viewdir;

    float ndotv                        = viewdir DOT mesh->normals[i];
    float                     sintheta = sqrt(1.0f - trimesh::sqr(ndotv));
    float                     csctheta = 1.0f / sintheta;
    float                     u = (viewdir DOT mesh->pdir1[i]) * csctheta;
    float                     v = (viewdir DOT mesh->pdir2[i]) * csctheta;
    float kr = mesh->curv1[i] * u * u + mesh->curv2[i] * v * v;
    float tr = u * v * (mesh->curv2[i] - mesh->curv1[i]);
    float kt =
        mesh->curv1[i] * (1.0f - u * u) + mesh->curv2[i] * (1.0f - v * v);
    trimesh::vec w           = u * mesh->pdir1[i] + v * mesh->pdir2[i];
    trimesh::vec wperp       = u * mesh->pdir2[i] - v * mesh->pdir1[i];
    const trimesh::Vec<4>& C = mesh->dcurv[i];

    trimesh::vec g = mesh->pdir1[i] *
                         (u * u * C[0] + 2.0f * u * v * C[1] + v * v * C[2]) +
                     mesh->pdir2[i] *
                         (u * u * C[1] + 2.0f * u * v * C[2] + v * v * C[3]) -
                     2.0f * csctheta * tr *
                         (rlen_viewdir * wperp + ndotv * (tr * w + kt * wperp));
    g *= (1.0f - trimesh::sqr(ndotv));
    g -= 2.0f * kr * sintheta * ndotv * (kr * w + tr * wperp);
    return g;
}

// Find a zero crossing between val0 and val1 by linear interpolation
// Returns 0 if zero crossing is at val0, 1 if at val1, etc.
static inline float find_zero_linear(float val0, float val1)
{
    return val0 / (val0 - val1);
}

// Find a zero crossing using Hermite interpolation
float find_zero_hermite(int v0, int v1, float val0, float val1,
                        const trimesh::vec& grad0, const trimesh::vec& grad1,
                        trimesh::TriMesh* mesh)
{
    if (val0 == val1)
        return 0.5f;

    // Find derivatives along edge (of interpolation parameter in [0,1]
    // which means that e01 doesn't get normalized)
    trimesh::vec e01 = mesh->vertices[v1] - mesh->vertices[v0];
    float d0 = e01 DOT grad0, d1 = e01 DOT grad1;

    // This next line would reduce val to linear interpolation
    // d0 = d1 = (val1 - val0);

    // Use hermite interpolation:
    //   val(s) = h1(s)*val0 + h2(s)*val1 + h3(s)*d0 + h4(s)*d1
    // where
    //  h1(s) = 2*s^3 - 3*s^2 + 1
    //  h2(s) = 3*s^2 - 2*s^3
    //  h3(s) = s^3 - 2*s^2 + s
    //  h4(s) = s^3 - s^2
    //
    //  val(s)  = [2(val0-val1) +d0+d1]*s^3 +
    //            [3(val1-val0)-2d0-d1]*s^2 + d0*s + val0
    // where
    //
    //  val(0) = val0; val(1) = val1; val'(0) = d0; val'(1) = d1
    //

    // Coeffs of cubic a*s^3 + b*s^2 + c*s + d
    float a = 2 * (val0 - val1) + d0 + d1;
    float b = 3 * (val1 - val0) - 2 * d0 - d1;
    float c = d0, d = val0;

    // -- Find a root by bisection
    // (as Newton can wander out of desired interval)

    // Start with entire [0,1] interval
    float sl = 0.0f, sr = 1.0f, valsl = val0, valsr = val1;

    // Check if we're in a (somewhat uncommon) 3-root situation, and pick
    // the middle root if it happens (given we aren't drawing curvy lines,
    // seems the best approach..)
    //
    // Find extrema of derivative (a -> 3a; b -> 2b, c -> c),
    // and check if they're both in [0,1] and have different signs
    float disc = 4 * b - 12 * a * c;
    if (disc > 0 && a != 0)
    {
        disc     = sqrt(disc);
        float r1 = (-2 * b + disc) / (6 * a);
        float r2 = (-2 * b - disc) / (6 * a);
        if (r1 >= 0 && r1 <= 1 && r2 >= 0 && r2 <= 1)
        {
            float vr1 = (((a * r1 + b) * r1 + c) * r1) + d;
            float vr2 = (((a * r2 + b) * r2 + c) * r2) + d;
            // When extrema have different signs inside an
            // interval with endpoints with different signs,
            // the middle root is in between the two extrema
            if ((vr1 < 0.0f && vr2 >= 0.0f) || (vr1 > 0.0f && vr2 <= 0.0f))
            {
                // 3 roots
                if (r1 < r2)
                {
                    sl    = r1;
                    valsl = vr1;
                    sr    = r2;
                }
                else
                {
                    sl    = r2;
                    valsl = vr2;
                    sr    = r1;
                }
            }
        }
    }

    // Bisection method (constant number of interations)
    for (int iter = 0; iter < 10; iter++)
    {
        float sbi    = (sl + sr) / 2.0f;
        float valsbi = (((a * sbi + b) * sbi) + c) * sbi + d;

        // Keep the half which has different signs
        if ((valsl < 0.0f && valsbi >= 0.0f) ||
            (valsl > 0.0f && valsbi <= 0.0f))
        {
            sr = sbi;
        }
        else
        {
            sl    = sbi;
            valsl = valsbi;
        }
    }

    return 0.5f * (sl + sr);
}

// Draw part of a zero-crossing curve on one triangle face, but only if
// "test_num/test_den" is positive.  v0,v1,v2 are the indices of the 3
// vertices, "val" are the values of the scalar field whose zero
// crossings we are finding, and "test_*" are the values we are testing
// to make sure they are positive.  This function assumes that val0 has
// opposite sign from val1 and val2 - the following function is the
// general one that figures out which one actually has the different sign.
void compute_face_isoline2(int v0, int v1, int v2, const isoline_params& params,
                           trimesh::TriMesh*             mesh,
                           const trimesh::point&         viewpos,
                           const trimesh::vec&           currcolor,
                           std::vector<trimesh::point3>& points,
                           std::vector<trimesh::vec4>&  colors)
{
    // How far along each edge?
    auto&& val      = params.val;
    auto&& test_num = params.test_num;
    auto&& test_den = params.test_den;
    float  w10      = params.do_hermite ?
                          find_zero_hermite(v0, v1, val[v0], val[v1],
                                            gradkr(v0, mesh, viewpos),
                                            gradkr(v1, mesh, viewpos), mesh) :
                          find_zero_linear(val[v0], val[v1]);
    float  w01      = 1.0f - w10;
    float  w20      = params.do_hermite ?
                          find_zero_hermite(v0, v2, val[v0], val[v2],
                                            gradkr(v0, mesh, viewpos),
                                            gradkr(v2, mesh, viewpos), mesh) :
                          find_zero_linear(val[v0], val[v2]);
    float  w02      = 1.0f - w20;

    // Points along edges
    trimesh::point p1 =
        w01 * mesh->vertices[v0] + w10 * mesh->vertices[v1];
    trimesh::point p2 =
        w02 * mesh->vertices[v0] + w20 * mesh->vertices[v2];

    float test_num1 = 1.0f, test_num2 = 1.0f;
    float test_den1 = 1.0f, test_den2 = 1.0f;
    float z1 = 0.0f, z2 = 0.0f;
    bool  valid1 = true;
    if (params.do_test)
    {
        // Interpolate to find value of test at p1, p2
        test_num1 = w01 * test_num[v0] + w10 * test_num[v1];
        test_num2 = w02 * test_num[v0] + w20 * test_num[v2];
        if (!test_den.empty())
        {
            test_den1 = w01 * test_den[v0] + w10 * test_den[v1];
            test_den2 = w02 * test_den[v0] + w20 * test_den[v2];
        }
        // First point is valid iff num1/den1 is positive,
        // i.e. the num and den have the same sign
        valid1 = ((test_num1 >= 0.0f) == (test_den1 >= 0.0f));
        // There are two possible zero crossings of the test,
        // corresponding to zeros of the num and den
        if ((test_num1 >= 0.0f) != (test_num2 >= 0.0f))
            z1 = test_num1 / (test_num1 - test_num2);
        if ((test_den1 >= 0.0f) != (test_den2 >= 0.0f))
            z2 = test_den1 / (test_den1 - test_den2);
        // Sort and order the zero crossings
        if (z1 == 0.0f)
            z1 = z2, z2 = 0.0f;
        else if (z2 < z1)
            std::swap(z1, z2);
    }

    // If the beginning of the segment was not valid, and
    // no zero crossings, then whole segment invalid
    if (!valid1 && !z1 && !z2)
        return;

    // Draw the valid piece(s)
    int npts = 0;
    if (valid1)
    {
        colors.push_back({currcolor[0], currcolor[1], currcolor[2],
                          test_num1 / (test_den1 * params.fade + test_num1)});
        points.push_back(p1);
        npts++;
    }
    if (z1)
    {
        float num = (1.0f - z1) * test_num1 + z1 * test_num2;
        float den = (1.0f - z1) * test_den1 + z1 * test_den2;
        colors.push_back({currcolor[0], currcolor[1], currcolor[2],
                  num / (den * params.fade + num)});
        points.push_back((1.0f - z1) * p1 + z1 * p2);
        npts++;
    }
    if (z2)
    {
        float num = (1.0f - z2) * test_num1 + z2 * test_num2;
        float den = (1.0f - z2) * test_den1 + z2 * test_den2;
        colors.push_back({currcolor[0], currcolor[1], currcolor[2],
                  num / (den * params.fade + num)});
        points.push_back((1.0f - z2) * p1 + z2 * p2);
        npts++;
    }
    if (npts != 2)
    {
        colors.push_back({currcolor[0], currcolor[1], currcolor[2],
                  test_num2 / (test_den2 * params.fade + test_num2)});
        points.push_back(p2);
    }
}

// See above.  This is the driver function that figures out which of
// v0, v1, v2 has a different sign from the others.
void compute_face_isoline(int v0, int v1, int v2, const isoline_params& params,
                          trimesh::TriMesh*             mesh,
                          const trimesh::point&         viewpos,
                          const trimesh::vec&           currcolor,
                          std::vector<trimesh::point3>& points,
                          std::vector<trimesh::vec4>&  colors)
{
    // Backface culling
    if (likely(params.do_bfcull && params.ndotv[v0] <= 0.0f &&
               params.ndotv[v1] <= 0.0f && params.ndotv[v2] <= 0.0f))
        return;

    // Quick reject if derivs are negative
    auto&& test_num = params.test_num;
    auto&& test_den = params.test_den;
    if (params.do_test)
    {
        if (test_den.empty())
        {
            if (test_num[v0] <= 0.0f && test_num[v1] <= 0.0f &&
                test_num[v2] <= 0.0f)
                return;
        }
        else
        {
            if (test_num[v0] <= 0.0f && test_den[v0] >= 0.0f &&
                test_num[v1] <= 0.0f && test_den[v1] >= 0.0f &&
                test_num[v2] <= 0.0f && test_den[v2] >= 0.0f)
                return;
            if (test_num[v0] >= 0.0f && test_den[v0] <= 0.0f &&
                test_num[v1] >= 0.0f && test_den[v1] <= 0.0f &&
                test_num[v2] >= 0.0f && test_den[v2] <= 0.0f)
                return;
        }
    }
    auto&& val = params.val;

    // Figure out which val has different sign, and draw
    if ((val[v0] < 0.0f && val[v1] >= 0.0f && val[v2] >= 0.0f) ||
        (val[v0] > 0.0f && val[v1] <= 0.0f && val[v2] <= 0.0f))
        compute_face_isoline2(v0, v1, v2, params, mesh, viewpos, currcolor,
                              points, colors);
    else if ((val[v1] < 0.0f && val[v2] >= 0.0f && val[v0] >= 0.0f) ||
             (val[v1] > 0.0f && val[v2] <= 0.0f && val[v0] <= 0.0f))
        compute_face_isoline2(v1, v2, v0, params, mesh, viewpos, currcolor,
                              points, colors);
    else if ((val[v2] < 0.0f && val[v0] >= 0.0f && val[v1] >= 0.0f) ||
             (val[v2] > 0.0f && val[v0] <= 0.0f && val[v1] <= 0.0f))
        compute_face_isoline2(v2, v0, v1, params, mesh, viewpos, currcolor,
                              points, colors);
}

// Takes a scalar field and renders the zero crossings, but only where
// test_num/test_den is greater than 0.
std::pair<std::vector<trimesh::point3>, std::vector<trimesh::vec4>>
compute_isolines(const isoline_params& params, trimesh::TriMesh* mesh,
                      const trimesh::point& viewpos,
                      const trimesh::vec&   currcolor)
{

    const int* t        = &mesh->tstrips[0];
    const int* stripend = t;
    const int* end      = t + mesh->tstrips.size();

    std::pair<std::vector<trimesh::point3>, std::vector<trimesh::vec4>> ret;
    // Walk through triangle strips
    while (1)
    {
        if (t >= stripend)
        {
            if (t >= end)
                return ret;
            // New strip: each strip is stored as
            // length followed by indices
            stripend = t + 1 + *t;
            // Skip over length plus first two indices of
            // first face
            t += 3;
        }
        // Draw a line if, among the values in this triangle,
        // at least one is positive and one is negative
        auto&&       val = params.val;
        const float &v0 = val[*t], &v1 = val[*(t - 1)], &v2 = val[*(t - 2)];
        if ((v0 > 0.0f || v1 > 0.0f || v2 > 0.0f) &&
            (v0 < 0.0f || v1 < 0.0f || v2 < 0.0f))
            compute_face_isoline(*(t - 2), *(t - 1), *t, params, mesh,
                                 viewpos, currcolor, ret.first, ret.second);
        t++;
    }
}

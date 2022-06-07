/*
Authors:
  Szymon Rusinkiewicz, Princeton University
  Doug DeCarlo, Rutgers University

With contributions by:
  Xiaofeng Mi, Rutgers University
  Tilke Judd, MIT

rtsc.cc
Real-time suggestive contours - these days, it also draws many other lines.
*/

#include "GL/glui.h"
#include "GL/glut.h"
#include "GLCamera.h"
#include "TriMesh.h"
#include "TriMesh_algo.h"
#include "XForm.h"
#include "apparentridge.h"
#include "isolines.h"
#include "timestamp.h"
#include <stdio.h>
#include <stdlib.h>
#ifndef DARWIN
//#include <GL/glext.h>
#endif
#include <algorithm>
#include <string>

// Set to false for hardware that has problems with display lists
const bool use_dlists = true;
// Set to false for hardware that has problems with supplying 3D texture coords
const bool use_3dtexc = false;

// Two cameras: the primary one, and an alternate one to fix the lines
// and see them from a different direction
int               dual_vpmode = false, mouse_moves_alt = false;
trimesh::GLCamera camera{}, camera_alt{};
trimesh::xform    xf{}, xf_alt{};
float             fov = 0.7f;
double            alt_projmatrix[16]{};
std::string             xffilename{}; // Filename where we look for "home" position
trimesh::point    viewpos{};    // Current view position
trimesh::TriMesh* glut_mesh = nullptr;

// Toggles for drawing various lines
int   draw_extsil = 0, draw_c = 1, draw_sc = 1;
int   draw_sh = 0, draw_phridges = 0, draw_phvalleys = 0;
int   draw_ridges = 0, draw_valleys = 0, draw_apparent = 0;
int   draw_K = 0, draw_H = 0, draw_DwKr = 0;
int   draw_bdy = 0, draw_isoph = 0, draw_topo = 0;
int   niso = 20, ntopo = 20;
float topo_offset = 0.0f;

// Toggles for tests we perform
int draw_hidden = 0;
int test_c = 1, test_sc = 1, test_sh = 1, test_ph = 1, test_rv = 1, test_ar = 1;
float sug_thresh = 0.01, sh_thresh = 0.02, ph_thresh = 0.04;
float rv_thresh = 0.1, ar_thresh = 0.1;

// Toggles for style
int use_texture = 0;
int draw_faded  = 1;
int draw_colors = 0;
int use_hermite = 0;

// Mesh colorization
enum
{
    COLOR_WHITE,
    COLOR_GRAY,
    COLOR_CURV,
    COLOR_GCURV,
    COLOR_MESH
};
const int                   ncolor_styles = 5;
int                         color_style   = COLOR_WHITE;
std::vector<trimesh::Color> curv_colors, gcurv_colors;
int                         draw_edges = false;

// Lighting
enum
{
    LIGHTING_NONE,
    LIGHTING_LAMBERTIAN,
    LIGHTING_LAMBERTIAN2,
    LIGHTING_HEMISPHERE,
    LIGHTING_TOON,
    LIGHTING_TOONBW,
    LIGHTING_GOOCH
};
const int      nlighting_styles = 7;
int            lighting_style   = LIGHTING_NONE;
GLUI_Rotation* lightdir         = NULL;
float lightdir_matrix[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
int   light_wrt_camera    = true;

// Per-vertex vectors
int draw_norm = 0, draw_curv1 = 0, draw_curv2 = 0, draw_asymp = 0;
int draw_w = 0, draw_wperp = 0;

// Other miscellaneous variables
float        feature_size; // Used to make thresholds dimensionless
float        currsmooth;   // Used in smoothing
trimesh::vec currcolor;    // Current line color

// Draw triangle strips.  They are stored as length followed by values.
void draw_tstrips(trimesh::TriMesh* mesh)
{
    const int* t   = &mesh->tstrips[0];
    const int* end = t + mesh->tstrips.size();
    while (likely(t < end))
    {
        int striplen = *t++;
        glDrawElements(GL_TRIANGLE_STRIP, striplen, GL_UNSIGNED_INT, t);
        t += striplen;
    }
}

// Create a texture with a black line of the given width.
void make_texture(float width)
{
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    int                   texsize  = 1024;
    static unsigned char* texture  = new unsigned char[texsize * texsize];
    int                   miplevel = 0;
    while (texsize)
    {
        for (int i = 0; i < texsize * texsize; i++)
        {
            float x   = (float)(i % texsize) - 0.5f * texsize + 0.5f;
            float y   = (float)(i / texsize) - 0.5f * texsize + 0.5f;
            float val = 1;
            if (texsize >= 4)
                if (fabs(x) < width && y > 0.0f)
                    val = trimesh::sqr(std::max(1.0f - y, 0.0f));
            texture[i] = std::min(std::max(int(256.0f * val), 0), 255);
        }
        glTexImage2D(GL_TEXTURE_2D, miplevel, GL_LUMINANCE, texsize, texsize, 0,
                     GL_LUMINANCE, GL_UNSIGNED_BYTE, texture);
        texsize >>= 1;
        miplevel++;
    }

    float bgcolor[] = {1, 1, 1, 1};
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, bgcolor);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR_MIPMAP_LINEAR);
#ifdef GL_EXT_texture_filter_anisotropic
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 1);
#endif
}

// Draw contours and suggestive contours using texture mapping
void draw_c_sc_texture(const std::vector<float>& ndotv,
                       const std::vector<float>& kr,
                       const std::vector<float>& sctest_num,
                       const std::vector<float>& sctest_den,
                       trimesh::TriMesh* mesh)
{
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, &mesh->vertices[0][0]);

    static std::vector<float> texcoords;
    int                       nv = mesh->vertices.size();
    texcoords.resize(2 * nv);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glTexCoordPointer(2, GL_FLOAT, 0, &texcoords[0]);

    // Remap texture coordinates from [-1..1] to [0..1]
    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();
    glTranslatef(0.5, 0.5, 0.0);
    glScalef(0.5, 0.5, 0.0);
    glMatrixMode(GL_MODELVIEW);

    float bgcolor[] = {1, 1, 1, 1};
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_BLEND);
    glTexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, bgcolor);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_DST_COLOR, GL_ZERO); // Multiplies texture into FB
    glEnable(GL_TEXTURE_2D);
    glDepthFunc(GL_LEQUAL);

    // First drawing pass for contours
    if (draw_c)
    {
        // Set up the texture for the contour pass
        static GLuint texcontext_c = 0;
        if (!texcontext_c)
        {
            glGenTextures(1, &texcontext_c);
            glBindTexture(GL_TEXTURE_2D, texcontext_c);
            make_texture(4.0);
        }
        glBindTexture(GL_TEXTURE_2D, texcontext_c);
        if (draw_colors)
            glColor3f(0.0, 0.6, 0.0);
        else
            glColor3f(0.05, 0.05, 0.05);

        // Compute texture coordinates and draw
        for (int i = 0; i < nv; i++)
        {
            texcoords[2 * i]     = ndotv[i];
            texcoords[2 * i + 1] = 0.5f;
        }
        draw_tstrips(mesh);
    }

    // Second drawing pass for suggestive contours.  This should eventually
    // be folded into the previous one with multitexturing.
    if (draw_sc)
    {
        static GLuint texcontext_sc = 0;
        if (!texcontext_sc)
        {
            glGenTextures(1, &texcontext_sc);
            glBindTexture(GL_TEXTURE_2D, texcontext_sc);
            make_texture(2.0);
        }
        glBindTexture(GL_TEXTURE_2D, texcontext_sc);
        if (draw_colors)
            glColor3f(0.0, 0.0, 0.8);
        else
            glColor3f(0.05, 0.05, 0.05);

        float feature_size2 = trimesh::sqr(feature_size);
        for (int i = 0; i < nv; i++)
        {
            texcoords[2 * i] = feature_size * kr[i];
            texcoords[2 * i + 1] =
                feature_size2 * sctest_num[i] / sctest_den[i];
        }
        draw_tstrips(mesh);
    }

    glDisable(GL_TEXTURE_2D);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
}

// Color the mesh by curvatures
void compute_curv_colors(trimesh::TriMesh* mesh)
{
    float cscale = trimesh::sqr(8.0f * feature_size);

    int nv = mesh->vertices.size();
    curv_colors.resize(nv);
    for (int i = 0; i < nv; i++)
    {
        float H = 0.5f * (mesh->curv1[i] + mesh->curv2[i]);
        float K = mesh->curv1[i] * mesh->curv2[i];
        float h = 4.0f / 3.0f * fabs(atan2(H * H - K, H * H * trimesh::sgn(H)));
        float s = M_2_PI * atan((2.0f * H * H - K) * cscale);
        curv_colors[i] = trimesh::Color::hsv(h, s, 1.0);
    }
}

// Similar, but grayscale mapping of mean curvature H
void compute_gcurv_colors(trimesh::TriMesh* mesh)
{
    float cscale = 10.0f * feature_size;

    int nv = mesh->vertices.size();
    gcurv_colors.resize(nv);
    for (int i = 0; i < nv; i++)
    {
        float H         = 0.5f * (mesh->curv1[i] + mesh->curv2[i]);
        float c         = (atan(H * cscale) + M_PI_2) / M_PI;
        c               = sqrt(c);
        int C           = int(std::min(std::max(256.0 * c, 0.0), 255.99));
        gcurv_colors[i] = trimesh::Color(C, C, C);
    }
}

// Set up textures to be used for the lighting.
// These are indexed by (n dot l), though they are actually 2D textures
// with a height of 1 because some hardware (cough, cough, ATI) is
// thoroughly broken for 1D textures...
void make_light_textures(GLuint* texture_contexts)
{
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    const int     texsize = 256;
    unsigned char texture[3 * texsize];

    glGenTextures(nlighting_styles, texture_contexts);

    // Simple diffuse shading
    glBindTexture(GL_TEXTURE_2D, texture_contexts[LIGHTING_LAMBERTIAN]);
    for (int i = 0; i < texsize; i++)
    {
        float z    = float(i + 1 - texsize / 2) / (0.5f * texsize);
        texture[i] = std::max(0, int(255 * z));
    }
    glTexImage2D(GL_TEXTURE_2D, 0, 1, texsize, 1, 0, GL_LUMINANCE,
                 GL_UNSIGNED_BYTE, texture);

    // Diffuse shading with gamma = 2
    glBindTexture(GL_TEXTURE_2D, texture_contexts[LIGHTING_LAMBERTIAN2]);
    for (int i = 0; i < texsize; i++)
    {
        float z    = float(i + 1 - texsize / 2) / (0.5f * texsize);
        texture[i] = std::max(0, int(255 * sqrt(z)));
    }
    glTexImage2D(GL_TEXTURE_2D, 0, 1, texsize, 1, 0, GL_LUMINANCE,
                 GL_UNSIGNED_BYTE, texture);

    // Lighting from a hemisphere of light
    glBindTexture(GL_TEXTURE_2D, texture_contexts[LIGHTING_HEMISPHERE]);
    for (int i = 0; i < texsize; i++)
    {
        float z    = float(i + 1 - texsize / 2) / (0.5f * texsize);
        texture[i] = std::max(0, int(255 * (0.5f + 0.5f * z)));
    }
    glTexImage2D(GL_TEXTURE_2D, 0, 1, texsize, 1, 0, GL_LUMINANCE,
                 GL_UNSIGNED_BYTE, texture);

    // A soft gray/white toon shader
    glBindTexture(GL_TEXTURE_2D, texture_contexts[LIGHTING_TOON]);
    for (int i = 0; i < texsize; i++)
    {
        float z    = float(i + 1 - texsize / 2) / (0.5f * texsize);
        int   tmp  = int(255 * z);
        texture[i] = std::min(std::max(2 * (tmp - 50), 210), 255);
    }
    glTexImage2D(GL_TEXTURE_2D, 0, 1, texsize, 1, 0, GL_LUMINANCE,
                 GL_UNSIGNED_BYTE, texture);

    // A hard black/white toon shader
    glBindTexture(GL_TEXTURE_2D, texture_contexts[LIGHTING_TOONBW]);
    for (int i = 0; i < texsize; i++)
    {
        float z    = float(i + 1 - texsize / 2) / (0.5f * texsize);
        int   tmp  = int(255 * z);
        texture[i] = std::min(std::max(25 * (tmp - 20), 0), 255);
    }
    glTexImage2D(GL_TEXTURE_2D, 0, 1, texsize, 1, 0, GL_LUMINANCE,
                 GL_UNSIGNED_BYTE, texture);

    // A Gooch-inspired yellow-to-blue color ramp
    glBindTexture(GL_TEXTURE_2D, texture_contexts[LIGHTING_GOOCH]);
    for (int i = 0; i < texsize; i++)
    {
        float z            = float(i + 1 - texsize / 2) / (0.5f * texsize);
        float r            = 0.75f + 0.25f * z;
        float g            = r;
        float b            = 0.9f - 0.1f * z;
        texture[3 * i]     = std::max(0, int(255 * r));
        texture[3 * i + 1] = std::max(0, int(255 * g));
        texture[3 * i + 2] = std::max(0, int(255 * b));
    }
    glTexImage2D(GL_TEXTURE_2D, 0, 3, texsize, 1, 0, GL_RGB, GL_UNSIGNED_BYTE,
                 texture);
}

// Draw the basic mesh, which we'll overlay with lines
void draw_base_mesh(trimesh::TriMesh* mesh)
{
    int nv = mesh->vertices.size();

    // Enable the vertex array
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, &mesh->vertices[0][0]);

    // Set up for color
    switch (color_style)
    {
    case COLOR_WHITE:
        glColor3f(1, 1, 1);
        break;
    case COLOR_GRAY:
        glColor3f(0.65, 0.65, 0.65);
        break;
    case COLOR_CURV:
        if (curv_colors.empty())
            compute_curv_colors(mesh);
        glEnableClientState(GL_COLOR_ARRAY);
        glColorPointer(3, GL_FLOAT, 0, &curv_colors[0][0]);
        break;
    case COLOR_GCURV:
        if (gcurv_colors.empty())
            compute_gcurv_colors(mesh);
        glEnableClientState(GL_COLOR_ARRAY);
        glColorPointer(3, GL_FLOAT, 0, &gcurv_colors[0][0]);
        break;
    case COLOR_MESH:
        glEnableClientState(GL_COLOR_ARRAY);
        glColorPointer(3, GL_FLOAT, 0, &mesh->colors[0][0]);
        break;
    }

    // Set up for lighting
    std::vector<float> ndotl;
    if (use_3dtexc)
    {
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
        glTexCoordPointer(3, GL_FLOAT, 0, &mesh->normals[0][0]);
    }
    if (lighting_style != LIGHTING_NONE)
    {
        // Set up texture
        static GLuint texture_contexts[nlighting_styles];
        static bool   havetextures = false;
        if (!havetextures)
        {
            make_light_textures(texture_contexts);
            havetextures = true;
        }

        // Compute lighting direction -- the Z axis from the widget
        trimesh::vec lightdir(&lightdir_matrix[8]);
        if (light_wrt_camera)
            lightdir = rot_only(inv(xf)) * lightdir;
        float rotamount =
            180.0f / M_PI * acos(lightdir DOT trimesh::vec(1, 0, 0));
        trimesh::vec rotaxis = lightdir CROSS trimesh::vec(1, 0, 0);

        // Texture matrix: remap from normals to texture coords
        glMatrixMode(GL_TEXTURE);
        glLoadIdentity();
        glTranslatef(0.5, 0.5, 0); // Remap [-0.5 .. 0.5] -> [0 .. 1]
        glScalef(0.496, 0, 0);     // Remap [-1 .. 1] -> (-0.5 .. 0.5)
        if (use_3dtexc)            // Rotate normals, else see below
            glRotatef(rotamount, rotaxis[0], rotaxis[1], rotaxis[2]);
        glMatrixMode(GL_MODELVIEW);

        // Bind and enable the texturing
        glBindTexture(GL_TEXTURE_2D, texture_contexts[lighting_style]);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
        glEnable(GL_TEXTURE_2D);

        // On broken hardware, compute 1D tex coords by hand
        if (!use_3dtexc)
        {
            ndotl.resize(nv);
            for (int i = 0; i < nv; i++)
                ndotl[i] = mesh->normals[i] DOT lightdir;
            glEnableClientState(GL_TEXTURE_COORD_ARRAY);
            glTexCoordPointer(1, GL_FLOAT, 0, &ndotl[0]);
        }
    }

    // Draw the mesh, possibly with color and/or lighting
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);
    glPolygonOffset(5.0f, 30.0f);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glEnable(GL_CULL_FACE);

    if (use_dlists && !glIsEnabled(GL_COLOR_ARRAY) &&
        (use_3dtexc || lighting_style == LIGHTING_NONE))
    {
        // Draw the geometry - using display list
        if (!glIsList(1))
        {
            glNewList(1, GL_COMPILE);
            draw_tstrips(mesh);
            glEndList();
        }
        glCallList(1);
    }
    else
    {
        // Draw geometry, no display list
        draw_tstrips(mesh);
    }

    // Reset everything
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisable(GL_CULL_FACE);
    glDisable(GL_TEXTURE_2D);
    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glDisable(GL_POLYGON_OFFSET_FILL);
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_FALSE); // Do not remove me, else get dotted lines

    // Draw the mesh edges on top, if requested
    glLineWidth(1);
    if (draw_edges)
    {
        glPolygonMode(GL_FRONT, GL_LINE);
        glColor3f(0.5, 1.0, 1.0);
        draw_tstrips(mesh);
        glPolygonMode(GL_FRONT, GL_FILL);
    }

    // Draw various per-vertex vectors, if requested
    float line_len = 0.5f * mesh->feature_size();
    if (draw_norm)
    {
        // Normals
        glColor3f(0.7, 0.7, 0);
        glBegin(GL_LINES);
        for (int i = 0; i < nv; i++)
        {
            glVertex3fv(mesh->vertices[i]);
            glVertex3fv(mesh->vertices[i] +
                        2.0f * line_len * mesh->normals[i]);
        }
        glEnd();
        glPointSize(3);
        glDrawArrays(GL_POINTS, 0, nv);
    }
    if (draw_curv1)
    {
        // Maximum-magnitude principal direction
        glColor3f(0.2, 0.7, 0.2);
        glBegin(GL_LINES);
        for (int i = 0; i < nv; i++)
        {
            glVertex3fv(mesh->vertices[i] - line_len * mesh->pdir1[i]);
            glVertex3fv(mesh->vertices[i] + line_len * mesh->pdir1[i]);
        }
        glEnd();
    }
    if (draw_curv2)
    {
        // Minimum-magnitude principal direction
        glColor3f(0.7, 0.2, 0.2);
        glBegin(GL_LINES);
        for (int i = 0; i < nv; i++)
        {
            glVertex3fv(mesh->vertices[i] - line_len * mesh->pdir2[i]);
            glVertex3fv(mesh->vertices[i] + line_len * mesh->pdir2[i]);
        }
        glEnd();
    }
    if (draw_asymp)
    {
        // Asymptotic directions, scaled by sqrt(-K)
        float ascale2 = trimesh::sqr(5.0f * line_len * feature_size);
        glColor3f(1, 0.5, 0);
        glBegin(GL_LINES);
        for (int i = 0; i < nv; i++)
        {
            const float& k1     = mesh->curv1[i];
            const float& k2     = mesh->curv2[i];
            float        scale2 = -k1 * k2 * ascale2;
            if (scale2 <= 0.0f)
                continue;
            trimesh::vec ax = sqrt(scale2 * k2 / (k2 - k1)) * mesh->pdir1[i];
            trimesh::vec ay = sqrt(scale2 * k1 / (k1 - k2)) * mesh->pdir2[i];
            glVertex3fv(mesh->vertices[i] + ax + ay);
            glVertex3fv(mesh->vertices[i] - ax - ay);
            glVertex3fv(mesh->vertices[i] + ax - ay);
            glVertex3fv(mesh->vertices[i] - ax + ay);
        }
        glEnd();
    }
    if (draw_w)
    {
        // Projected view direction
        glColor3f(0, 0, 1);
        glBegin(GL_LINES);
        for (int i = 0; i < nv; i++)
        {
            trimesh::vec w = viewpos - mesh->vertices[i];
            w -= mesh->normals[i] * (w DOT mesh->normals[i]);
            normalize(w);
            glVertex3fv(mesh->vertices[i]);
            glVertex3fv(mesh->vertices[i] + line_len * w);
        }
        glEnd();
    }
    if (draw_wperp)
    {
        // Perpendicular to projected view direction
        glColor3f(0, 0, 1);
        glBegin(GL_LINES);
        for (int i = 0; i < nv; i++)
        {
            trimesh::vec w = viewpos - mesh->vertices[i];
            w -= mesh->normals[i] * (w DOT mesh->normals[i]);
            trimesh::vec wperp = mesh->normals[i] CROSS w;
            normalize(wperp);
            glVertex3fv(mesh->vertices[i]);
            glVertex3fv(mesh->vertices[i] + line_len * wperp);
        }
        glEnd();
    }

    glDisableClientState(GL_VERTEX_ARRAY);
}

// Compute per-vertex n dot l, n dot v, radial curvature, and
// derivative of curvature for the current view
void compute_perview(std::vector<float>& ndotv, std::vector<float>& kr,
                     std::vector<float>& sctest_num,
                     std::vector<float>& sctest_den,
                     std::vector<float>& shtest_num, std::vector<float>& q1,
                     std::vector<trimesh::vec2>& t1, std::vector<float>& Dt1q1,
                     bool extra_sin2theta,
                     trimesh::TriMesh* mesh)
{
    if (!mesh)
        return;

    if (draw_apparent)
        mesh->need_adjacentfaces();

    int nv = mesh->vertices.size();

    float scthresh  = sug_thresh / trimesh::sqr(feature_size);
    float shthresh  = sh_thresh / trimesh::sqr(feature_size);
    bool  need_DwKr = (draw_sc || draw_sh || draw_DwKr);

    ndotv.resize(nv);
    kr.resize(nv);
    if (draw_apparent)
    {
        q1.resize(nv);
        t1.resize(nv);
        Dt1q1.resize(nv);
    }
    if (need_DwKr)
    {
        sctest_num.resize(nv);
        sctest_den.resize(nv);
        if (draw_sh)
            shtest_num.resize(nv);
    }

    // Compute quantities at each vertex
#pragma omp parallel for
    for (int i = 0; i < nv; i++)
    {
        // Compute n DOT v
        trimesh::vec viewdir = viewpos - mesh->vertices[i];
        float        rlv     = 1.0f / trimesh::len(viewdir);
        viewdir *= rlv;
        ndotv[i] = viewdir DOT mesh->normals[i];

        float u = viewdir DOT mesh->pdir1[i], u2 = u * u;
        float v = viewdir DOT mesh->pdir2[i], v2 = v * v;

        // Note:  this is actually Kr * sin^2 theta
        kr[i] = mesh->curv1[i] * u2 + mesh->curv2[i] * v2;

        if (draw_apparent)
        {
            float csc2theta = 1.0f / (u2 + v2);
            compute_viewdep_curv(mesh, i, ndotv[i], u2 * csc2theta,
                                 u * v * csc2theta, v2 * csc2theta, q1[i],
                                 t1[i]);
        }
        if (!need_DwKr)
            continue;

        // Use DwKr * sin(theta) / cos(theta) for cutoff test
        sctest_num[i] =
            u2 * (u * mesh->dcurv[i][0] + 3.0f * v * mesh->dcurv[i][1]) +
            v2 * (3.0f * u * mesh->dcurv[i][2] + v * mesh->dcurv[i][3]);
        float csc2theta = 1.0f / (u2 + v2);
        sctest_num[i] *= csc2theta;
        float tr = (mesh->curv2[i] - mesh->curv1[i]) * u * v * csc2theta;
        sctest_num[i] -= 2.0f * ndotv[i] * trimesh::sqr(tr);
        if (extra_sin2theta)
            sctest_num[i] *= u2 + v2;

        sctest_den[i] = ndotv[i];

        if (draw_sh)
        {
            shtest_num[i] = -sctest_num[i];
            shtest_num[i] -= shthresh * sctest_den[i];
        }
        sctest_num[i] -= scthresh * sctest_den[i];
    }
    if (draw_apparent)
    {
#pragma omp parallel for
        for (int i = 0; i < nv; i++)
            compute_Dt1q1(mesh, i, ndotv[i], q1, t1, Dt1q1[i]);
    }
}

// Draw part of a ridge/valley curve on one triangle face.  v0,v1,v2
// are the indices of the 3 vertices; this function assumes that the
// curve connects points on the edges v0-v1 and v1-v2
// (or connects point on v0-v1 to center if to_center is true)
void draw_segment_ridge(int v0, int v1, int v2, float emax0, float emax1,
                        float emax2, float kmax0, float kmax1, float kmax2,
                        float thresh, bool to_center,
                        trimesh::TriMesh* mesh)
{
    // Interpolate to find ridge/valley line segment endpoints
    // in this triangle and the curvatures there
    float          w10 = fabs(emax0) / (fabs(emax0) + fabs(emax1));
    float          w01 = 1.0f - w10;
    trimesh::point p01 =
        w01 * mesh->vertices[v0] + w10 * mesh->vertices[v1];
    float k01 = fabs(w01 * kmax0 + w10 * kmax1);

    trimesh::point p12;
    float          k12;
    if (to_center)
    {
        // Connect first point to center of triangle
        p12 = (mesh->vertices[v0] + mesh->vertices[v1] +
               mesh->vertices[v2]) /
              3.0f;
        k12 = fabs(kmax0 + kmax1 + kmax2) / 3.0f;
    }
    else
    {
        // Connect first point to second one (on next edge)
        float w21 = fabs(emax1) / (fabs(emax1) + fabs(emax2));
        float w12 = 1.0f - w21;
        p12       = w12 * mesh->vertices[v1] + w21 * mesh->vertices[v2];
        k12       = fabs(w12 * kmax1 + w21 * kmax2);
    }

    // Don't draw below threshold
    k01 -= thresh;
    if (k01 < 0.0f)
        k01 = 0.0f;
    k12 -= thresh;
    if (k12 < 0.0f)
        k12 = 0.0f;

    // Skip lines that you can't see...
    if (k01 == 0.0f && k12 == 0.0f)
        return;

    // Fade lines
    if (draw_faded)
    {
        k01 /= (k01 + thresh);
        k12 /= (k12 + thresh);
    }
    else
    {
        k01 = k12 = 1.0f;
    }

    // Draw the line segment
    glColor4f(currcolor[0], currcolor[1], currcolor[2], k01);
    glVertex3fv(p01);
    glColor4f(currcolor[0], currcolor[1], currcolor[2], k12);
    glVertex3fv(p12);
}

// Draw ridges or valleys (depending on do_ridge) in a triangle v0,v1,v2
// - uses ndotv for backface culling (enabled with do_bfcull)
// - do_test checks for curvature maxima/minina for ridges/valleys
//   (when off, it draws positive minima and negative maxima)
// Note: this computes ridges/valleys every time, instead of once at the
//   start (given they aren't view dependent, this is wasteful)
// Algorithm based on formulas of Ohtake et al., 2004.
void draw_face_ridges(int v0, int v1, int v2, bool do_ridge,
                      const std::vector<float>& ndotv, bool do_bfcull,
                      bool do_test, float thresh,
                      trimesh::TriMesh* mesh)
{
    // Backface culling
    if (do_bfcull && ndotv[v0] <= 0.0f && ndotv[v1] <= 0.0f &&
        ndotv[v2] <= 0.0f)
        return;

    // Check if ridge possible at vertices just based on curvatures
    if (do_ridge)
    {
        if ((mesh->curv1[v0] <= 0.0f) || (mesh->curv1[v1] <= 0.0f) ||
            (mesh->curv1[v2] <= 0.0f))
            return;
    }
    else
    {
        if ((mesh->curv1[v0] >= 0.0f) || (mesh->curv1[v1] >= 0.0f) ||
            (mesh->curv1[v2] >= 0.0f))
            return;
    }

    // Sign of curvature on ridge/valley
    float rv_sign = do_ridge ? 1.0f : -1.0f;

    // The "tmax" are the principal directions of maximal curvature,
    // flipped to point in the direction in which the curvature
    // is increasing (decreasing for valleys).  Note that this
    // is a bit different from the notation in Ohtake et al.,
    // but the tests below are equivalent.
    const float& emax0 = mesh->dcurv[v0][0];
    const float& emax1 = mesh->dcurv[v1][0];
    const float& emax2 = mesh->dcurv[v2][0];
    trimesh::vec tmax0 = rv_sign * mesh->dcurv[v0][0] * mesh->pdir1[v0];
    trimesh::vec tmax1 = rv_sign * mesh->dcurv[v1][0] * mesh->pdir1[v1];
    trimesh::vec tmax2 = rv_sign * mesh->dcurv[v2][0] * mesh->pdir1[v2];

    // We have a "zero crossing" if the tmaxes along an edge
    // point in opposite directions
    bool z01 = ((tmax0 DOT tmax1) <= 0.0f);
    bool z12 = ((tmax1 DOT tmax2) <= 0.0f);
    bool z20 = ((tmax2 DOT tmax0) <= 0.0f);

    if (z01 + z12 + z20 < 2)
        return;

    if (do_test)
    {
        const trimesh::point &p0 = mesh->vertices[v0],
                             &p1 = mesh->vertices[v1],
                             &p2 = mesh->vertices[v2];

        // Check whether we have the correct flavor of extremum:
        // Is the curvature increasing along the edge?
        z01 = z01 &&
              ((tmax0 DOT(p1 - p0)) >= 0.0f || (tmax1 DOT(p1 - p0)) <= 0.0f);
        z12 = z12 &&
              ((tmax1 DOT(p2 - p1)) >= 0.0f || (tmax2 DOT(p2 - p1)) <= 0.0f);
        z20 = z20 &&
              ((tmax2 DOT(p0 - p2)) >= 0.0f || (tmax0 DOT(p0 - p2)) <= 0.0f);

        if (z01 + z12 + z20 < 2)
            return;
    }

    // Draw line segment
    const float& kmax0 = mesh->curv1[v0];
    const float& kmax1 = mesh->curv1[v1];
    const float& kmax2 = mesh->curv1[v2];
    if (!z01)
    {
        draw_segment_ridge(v1, v2, v0, emax1, emax2, emax0, kmax1, kmax2, kmax0,
                           thresh, false, mesh);
    }
    else if (!z12)
    {
        draw_segment_ridge(v2, v0, v1, emax2, emax0, emax1, kmax2, kmax0, kmax1,
                           thresh, false, mesh);
    }
    else if (!z20)
    {
        draw_segment_ridge(v0, v1, v2, emax0, emax1, emax2, kmax0, kmax1, kmax2,
                           thresh, false, mesh);
    }
    else
    {
        // All three edges have crossings -- connect all to center
        draw_segment_ridge(v1, v2, v0, emax1, emax2, emax0, kmax1, kmax2, kmax0,
                           thresh, true, mesh);
        draw_segment_ridge(v2, v0, v1, emax2, emax0, emax1, kmax2, kmax0, kmax1,
                           thresh, true, mesh);
        draw_segment_ridge(v0, v1, v2, emax0, emax1, emax2, kmax0, kmax1, kmax2,
                           thresh, true, mesh);
    }
}

// Draw the ridges (valleys) of the mesh
void draw_mesh_ridges(bool do_ridge, const std::vector<float>& ndotv,
                      bool do_bfcull, bool do_test, float thresh,trimesh::TriMesh* mesh)
{
    const int* t        = &mesh->tstrips[0];
    const int* stripend = t;
    const int* end      = t + mesh->tstrips.size();

    // Walk through triangle strips
    while (1)
    {
        if (unlikely(t >= stripend))
        {
            if (unlikely(t >= end))
                return;
            // New strip: each strip is stored as
            // length followed by indices
            stripend = t + 1 + *t;
            // Skip over length plus first two indices of
            // first face
            t += 3;
        }

        draw_face_ridges(*(t - 2), *(t - 1), *t, do_ridge, ndotv, do_bfcull,
                         do_test, thresh, mesh);
        t++;
    }
}

// Draw principal highlights on a face
void draw_face_ph(int v0, int v1, int v2, bool do_ridge,
                  const std::vector<float>& ndotv, bool do_bfcull, bool do_test,
                  float thresh,trimesh::TriMesh* mesh)
{
    // Backface culling
    if (likely(do_bfcull && ndotv[v0] <= 0.0f && ndotv[v1] <= 0.0f &&
               ndotv[v2] <= 0.0f))
        return;

    // Orient principal directions based on the largest principal curvature
    float k0 = mesh->curv1[v0];
    float k1 = mesh->curv1[v1];
    float k2 = mesh->curv1[v2];
    if (do_test && do_ridge && std::min(std::min(k0, k1), k2) < 0.0f)
        return;
    if (do_test && !do_ridge && std::max(std::max(k0, k1), k2) > 0.0f)
        return;

    trimesh::vec d0   = mesh->pdir1[v0];
    trimesh::vec d1   = mesh->pdir1[v1];
    trimesh::vec d2   = mesh->pdir1[v2];

    // dref is the e1 vector with the largest |k1|
    trimesh::vec dref = d0;

    // Flip all the e1 to agree with dref
    if ((d0 DOT dref) < 0.0f)
        d0 = -d0;
    if ((d1 DOT dref) < 0.0f)
        d1 = -d1;
    if ((d2 DOT dref) < 0.0f)
        d2 = -d2;

    // If directions have flipped (more than 45 degrees), then give up
    if ((d0 DOT dref) < M_SQRT1_2 || (d1 DOT dref) < M_SQRT1_2 ||
        (d2 DOT dref) < M_SQRT1_2)
        return;

    // Compute view directions, dot products @ each vertex
    trimesh::vec viewdir0 = viewpos - mesh->vertices[v0];
    trimesh::vec viewdir1 = viewpos - mesh->vertices[v1];
    trimesh::vec viewdir2 = viewpos - mesh->vertices[v2];

    // Normalize these for cos(theta) later...
    trimesh::normalize(viewdir0);
    trimesh::normalize(viewdir1);
    trimesh::normalize(viewdir2);

    // e1 DOT w sin(theta)
    // -- which is zero when looking down e2
    float dot0 = viewdir0 DOT d0;
    float dot1 = viewdir1 DOT d1;
    float dot2 = viewdir2 DOT d2;

    // We have a "zero crossing" if the dot products along an edge
    // have opposite signs
    int z01 = (dot0 * dot1 <= 0.0f);
    int z12 = (dot1 * dot2 <= 0.0f);
    int z20 = (dot2 * dot0 <= 0.0f);

    if (z01 + z12 + z20 < 2)
        return;

    // Draw line segment
    float test0 =
        (trimesh::sqr(mesh->curv1[v0]) - trimesh::sqr(mesh->curv2[v0])) *
        viewdir0 DOT mesh->normals[v0];
    float            test1 =
        (trimesh::sqr(mesh->curv1[v1]) - trimesh::sqr(mesh->curv2[v1])) *
        viewdir1 DOT mesh->normals[v1];
    float            test2 =
        (trimesh::sqr(mesh->curv1[v2]) - trimesh::sqr(mesh->curv2[v2])) *
        viewdir2 DOT mesh->normals[v2];

    if (!z01)
    {
        draw_segment_ridge(v1, v2, v0, dot1, dot2, dot0, test1, test2, test0,
                           thresh, false, mesh);
    }
    else if (!z12)
    {
        draw_segment_ridge(v2, v0, v1, dot2, dot0, dot1, test2, test0, test1,
                           thresh, false, mesh);
    }
    else if (!z20)
    {
        draw_segment_ridge(v0, v1, v2, dot0, dot1, dot2, test0, test1, test2,
                           thresh, false, mesh);
    }
}

// Draw principal highlights
void draw_mesh_ph(bool do_ridge, const std::vector<float>& ndotv,
                  bool do_bfcull, bool do_test, float thresh,trimesh::TriMesh* mesh)
{
    const int* t        = &mesh->tstrips[0];
    const int* stripend = t;
    const int* end      = t + mesh->tstrips.size();

    // Walk through triangle strips
    while (1)
    {
        if (unlikely(t >= stripend))
        {
            if (unlikely(t >= end))
                return;
            // New strip: each strip is stored as
            // length followed by indices
            stripend = t + 1 + *t;
            // Skip over length plus first two indices of
            // first face
            t += 3;
        }

        draw_face_ph(*(t - 2), *(t - 1), *t, do_ridge, ndotv, do_bfcull,
                     do_test, thresh, mesh);
        t++;
    }
}

// Draw exterior silhouette of the mesh: this just draws
// thick contours, which are partially hidden by the mesh.
// Note: this needs to happen *before* draw_base_mesh...
void draw_silhouette(const std::vector<float>& ndotv, trimesh::TriMesh* mesh)
{
    glDepthMask(GL_FALSE);

    currcolor = trimesh::vec(0.0, 0.0, 0.0);
    isoline_params params{.val{ndotv}, .ndotv{ndotv}};

    glLineWidth(6);
    glBegin(GL_LINES);
    draw_isolines(params, mesh, viewpos, currcolor);
    glEnd();

    // Wide lines are gappy, so fill them in
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glPointSize(6);
    glBegin(GL_POINTS);
    draw_isolines(params, mesh, viewpos, currcolor);
    glEnd();

    glDisable(GL_POINT_SMOOTH);
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
}

// Draw the boundaries on the mesh
void draw_boundaries(bool do_hidden, trimesh::TriMesh* mesh)
{
    mesh->need_faces();
    mesh->need_across_edge();
    if (do_hidden)
    {
        glColor3f(0.6, 0.6, 0.6);
        glLineWidth(1.5);
    }
    else
    {
        glColor3f(0.05, 0.05, 0.05);
        glLineWidth(2.5);
    }
    glBegin(GL_LINES);
    for (size_t i = 0; i < mesh->faces.size(); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            if (mesh->across_edge[i][j] >= 0)
                continue;
            int v1 = mesh->faces[i][(j + 1) % 3];
            int v2 = mesh->faces[i][(j + 2) % 3];
            glVertex3fv(mesh->vertices[v1]);
            glVertex3fv(mesh->vertices[v2]);
        }
    }
    glEnd();
}

// Draw lines of n.l = const.
void draw_isophotes(const std::vector<float>& ndotv, trimesh::TriMesh* mesh)
{
    // Light direction
    trimesh::vec lightdir(&lightdir_matrix[8]);
    if (light_wrt_camera)
        lightdir = rot_only(inv(xf)) * lightdir;

    // Compute N dot L
    int                       nv = mesh->vertices.size();
    static std::vector<float> ndotl;
    ndotl.resize(nv);
    for (int i = 0; i < nv; i++)
        ndotl[i] = mesh->normals[i] DOT lightdir;

    if (draw_colors)
        currcolor = trimesh::vec(0.4, 0.8, 0.4);
    else
        currcolor = trimesh::vec(0.6, 0.6, 0.6);
    glColor3fv(currcolor);

    float dt = 1.0f / niso;
    for (int it = 0; it < niso; it++)
    {
        if (it == 0)
        {
            glLineWidth(2);
        }
        else
        {
            glLineWidth(1);
            for (int i = 0; i < nv; i++)
                ndotl[i] -= dt;
        }
        glBegin(GL_LINES);
        isoline_params params{.val{ndotl}, .ndotv{ndotv}, .do_bfcull = true};
        draw_isolines(params, mesh, viewpos, currcolor);
        glEnd();
    }

    // Draw negative isophotes (useful when light is not at camera)
    if (draw_colors)
        currcolor = trimesh::vec(0.6, 0.9, 0.6);
    else
        currcolor = trimesh::vec(0.7, 0.7, 0.7);
    glColor3fv(currcolor);

    for (int i = 0; i < nv; i++)
        ndotl[i] += dt * (niso - 1);
    for (int it = 1; it < niso; it++)
    {
        glLineWidth(1.0);
        for (int i = 0; i < nv; i++)
            ndotl[i] += dt;
        glBegin(GL_LINES);
        isoline_params params{.val{ndotl}, .ndotv{ndotv}, .do_bfcull = true};
        draw_isolines(params, mesh, viewpos, currcolor);
        glEnd();
    }
}

// Draw lines of constant depth
void draw_topolines(const std::vector<float>& ndotv, trimesh::TriMesh* mesh)
{
    // Camera direction and scale
    trimesh::vec camdir(xf[2], xf[6], xf[10]);
    float        depth_scale  = 0.5f / mesh->bsphere.r * ntopo;
    float        depth_offset = 0.5f * ntopo - topo_offset;

    // Compute depth
    static std::vector<float> depth;
    int                       nv = mesh->vertices.size();
    depth.resize(nv);
    for (int i = 0; i < nv; i++)
    {
        depth[i] =
            ((mesh->vertices[i] - mesh->bsphere.center) DOT camdir) *
                depth_scale +
            depth_offset;
    }

    // Draw the topo lines
    glLineWidth(1);
    glColor3f(0.5, 0.5, 0.5);
    for (int it = 0; it < ntopo; it++)
    {
        glBegin(GL_LINES);
        isoline_params params{.val{depth}, .do_bfcull = true};
        draw_isolines(params, mesh, viewpos, currcolor);
        glEnd();
        for (int i = 0; i < nv; i++)
            depth[i] -= 1.0f;
    }
}

// Draw K=0, H=0, and DwKr=thresh lines
void draw_misc(const std::vector<float>& ndotv, const std::vector<float>& DwKr,
               bool do_hidden, trimesh::TriMesh* mesh)
{
    if (do_hidden)
    {
        currcolor = trimesh::vec(1, 0.5, 0.5);
        glLineWidth(1);
    }
    else
    {
        currcolor = trimesh::vec(1, 0, 0);
        glLineWidth(2);
    }

    int nv = mesh->vertices.size();
    if (draw_K)
    {
        std::vector<float> K(nv);
        for (int i = 0; i < nv; i++)
            K[i] = mesh->curv1[i] * mesh->curv2[i];
        glBegin(GL_LINES);
        draw_isolines({.val{K}, .ndotv{ndotv}, .do_bfcull = !do_hidden},
                      mesh, viewpos, currcolor);
        glEnd();
    }
    if (draw_H)
    {
        std::vector<float> H(nv);
        for (int i = 0; i < nv; i++)
            H[i] = 0.5f * (mesh->curv1[i] + mesh->curv2[i]);
        glBegin(GL_LINES);
        draw_isolines({.val{H}, .ndotv{ndotv}, .do_bfcull = !do_hidden},
                      mesh, viewpos, currcolor);
        glEnd();
    }
    if (draw_DwKr)
    {
        glBegin(GL_LINES);
        draw_isolines({.val{DwKr}, .ndotv{ndotv}, .do_bfcull = !do_hidden},
                      mesh, viewpos, currcolor);
        glEnd();
    }
}

// Draw the mesh, possibly including a bunch of lines
void draw_mesh(trimesh::TriMesh* mesh)
{
    // These are static so the memory isn't reallocated on every frame
    static std::vector<float>         ndotv, kr;
    static std::vector<float>         sctest_num, sctest_den, shtest_num;
    static std::vector<float>         q1, Dt1q1;
    static std::vector<trimesh::vec2> t1;
    compute_perview(ndotv, kr, sctest_num, sctest_den, shtest_num, q1, t1,
                    Dt1q1, use_texture, mesh);

    // Enable antialiased lines
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Exterior silhouette
    if (draw_extsil)
        draw_silhouette(ndotv, mesh);

    // The mesh itself, possibly colored and/or lit
    glDisable(GL_BLEND);
    draw_base_mesh(mesh);
    glEnable(GL_BLEND);

    // Draw the lines on top

    // First rendering pass (in light gray) if drawing hidden lines
    if (draw_hidden)
    {
        glDisable(GL_DEPTH_TEST);

        // K=0, H=0, DwKr=thresh
        draw_misc(ndotv, sctest_num, true, mesh);

        // Apparent ridges
        if (draw_apparent)
        {
            if (draw_colors)
            {
                currcolor = trimesh::vec(0.8, 0.8, 0.4);
            }
            else
            {
                if (color_style == COLOR_GRAY ||
                    lighting_style != LIGHTING_NONE)
                    currcolor = trimesh::vec(0.75, 0.75, 0.75);
                else
                    currcolor = trimesh::vec(0.55, 0.55, 0.55);
            }
            if (draw_colors)
                glLineWidth(2);
            glBegin(GL_LINES);
            draw_mesh_app_ridges(mesh, ndotv, q1, t1, Dt1q1, true, test_ar,
                                 ar_thresh / trimesh::sqr(feature_size));
            glEnd();
        }

        // Ridges and valleys
        currcolor = trimesh::vec(0.55, 0.55, 0.55);
        if (draw_ridges)
        {
            if (draw_colors)
                currcolor = trimesh::vec(0.72, 0.6, 0.72);
            glLineWidth(1);
            glBegin(GL_LINES);
            draw_mesh_ridges(true, ndotv, false, test_rv,
                             rv_thresh / feature_size, mesh);
            glEnd();
        }
        if (draw_valleys)
        {
            if (draw_colors)
                currcolor = trimesh::vec(0.8, 0.72, 0.68);
            glLineWidth(1);
            glBegin(GL_LINES);
            draw_mesh_ridges(false, ndotv, false, test_rv,
                             rv_thresh / feature_size, mesh);
            glEnd();
        }

        // Principal highlights
        if (draw_phridges || draw_phvalleys)
        {
            if (draw_colors)
            {
                currcolor = trimesh::vec(0.5, 0, 0);
            }
            else
            {
                if (color_style == COLOR_GRAY ||
                    lighting_style != LIGHTING_NONE)
                    currcolor = trimesh::vec(0.75, 0.75, 0.75);
                else
                    currcolor = trimesh::vec(0.55, 0.55, 0.55);
            }
            glLineWidth(2);
            glBegin(GL_LINES);
            float thresh = ph_thresh / trimesh::sqr(feature_size);
            if (draw_phridges)
                draw_mesh_ph(true, ndotv, false, test_ph, thresh, mesh);
            if (draw_phvalleys)
                draw_mesh_ph(false, ndotv, false, test_ph, thresh, mesh);
            glEnd();
        }

        // Suggestive highlights
        if (draw_sh)
        {
            if (draw_colors)
            {
                currcolor = trimesh::vec(0.5, 0, 0);
            }
            else
            {
                if (color_style == COLOR_GRAY ||
                    lighting_style != LIGHTING_NONE)
                    currcolor = trimesh::vec(0.75, 0.75, 0.75);
                else
                    currcolor = trimesh::vec(0.55, 0.55, 0.55);
            }
            float fade = draw_faded ? 0.03f / trimesh::sqr(feature_size) : 0.0f;
            glLineWidth(2.5);
            glBegin(GL_LINES);
            draw_isolines({kr, shtest_num, sctest_den, ndotv, false,
                           !!use_hermite, !!test_sh, fade},
                          mesh, viewpos, currcolor);
            glEnd();
        }

        // Suggestive contours and contours
        if (draw_sc)
        {
            float fade = (draw_faded && test_sc) ?
                             0.03f / trimesh::sqr(feature_size) :
                             0.0f;
            if (draw_colors)
                currcolor = trimesh::vec(0.5, 0.5, 1.0);
            glLineWidth(1.5);
            glBegin(GL_LINES);
            draw_isolines({kr, sctest_num, sctest_den, ndotv, false,
                           !!use_hermite, !!test_sc, fade},
                          mesh, viewpos, currcolor);
            glEnd();
        }

        if (draw_c)
        {
            if (draw_colors)
                currcolor = trimesh::vec(0.4, 0.8, 0.4);
            glLineWidth(1.5);
            glBegin(GL_LINES);
            isoline_params params{
                .val{ndotv}, .test_num{kr}, .ndotv{ndotv}, .do_test = !!test_c};
            draw_isolines(params, mesh, viewpos, currcolor);
            glEnd();
        }

        // Boundaries
        if (draw_bdy)
            draw_boundaries(true, mesh);

        glEnable(GL_DEPTH_TEST);
    }

    // The main rendering pass

    // Isophotes
    if (draw_isoph)
        draw_isophotes(ndotv, mesh);

    // Topo lines
    if (draw_topo)
        draw_topolines(ndotv, mesh);

    // K=0, H=0, DwKr=thresh
    draw_misc(ndotv, sctest_num, false, mesh);

    // Apparent ridges
    currcolor = trimesh::vec(0.0, 0.0, 0.0);
    if (draw_apparent)
    {
        if (draw_colors)
            currcolor = trimesh::vec(0.4, 0.4, 0);
        glLineWidth(2.5);
        glBegin(GL_LINES);
        draw_mesh_app_ridges(mesh, ndotv, q1, t1, Dt1q1, true, test_ar,
                             ar_thresh / trimesh::sqr(feature_size));
        glEnd();
    }

    // Ridges and valleys
    currcolor    = trimesh::vec(0.0, 0.0, 0.0);
    
    if (draw_ridges)
    {
        if (draw_colors)
            currcolor = trimesh::vec(0.3, 0.0, 0.3);
        glLineWidth(2);
        glBegin(GL_LINES);
        draw_mesh_ridges(true, ndotv, true, test_rv, rv_thresh / feature_size, mesh);
        glEnd();
    }
    if (draw_valleys)
    {
        if (draw_colors)
            currcolor = trimesh::vec(0.5, 0.3, 0.2);
        glLineWidth(2);
        glBegin(GL_LINES);
        draw_mesh_ridges(false, ndotv, true, test_rv, rv_thresh / feature_size, mesh);
        glEnd();
    }

    // Principal highlights
    if (draw_phridges || draw_phvalleys)
    {
        if (draw_colors)
        {
            currcolor = trimesh::vec(0.5, 0, 0);
        }
        else
        {
            if (color_style == COLOR_GRAY || lighting_style != LIGHTING_NONE)
                currcolor = trimesh::vec(1, 1, 1);
            else
                currcolor = trimesh::vec(0, 0, 0);
        }
        glLineWidth(2);
        glBegin(GL_LINES);
        float thresh = ph_thresh / trimesh::sqr(feature_size);
        if (draw_phridges)
            draw_mesh_ph(true, ndotv, true, test_ph, thresh, mesh);
        if (draw_phvalleys)
            draw_mesh_ph(false, ndotv, true, test_ph, thresh, mesh);
        glEnd();
        currcolor = trimesh::vec(0.0, 0.0, 0.0);
    }

    // Suggestive highlights
    if (draw_sh)
    {
        if (draw_colors)
        {
            currcolor = trimesh::vec(0.5, 0, 0);
        }
        else
        {
            if (color_style == COLOR_GRAY || lighting_style != LIGHTING_NONE)
                currcolor = trimesh::vec(1.0, 1.0, 1.0);
            else
                currcolor = trimesh::vec(0.3, 0.3, 0.3);
        }
        float fade = draw_faded ? 0.03f / trimesh::sqr(feature_size) : 0.0f;
        glLineWidth(2.5);
        glBegin(GL_LINES);
        draw_isolines({kr, shtest_num, sctest_den, ndotv, true, !!use_hermite,
                       !!test_sh, fade},
                      mesh, viewpos, currcolor);
        glEnd();
        currcolor = trimesh::vec(0.0, 0.0, 0.0);
    }

    // Kr = 0 loops
    if (draw_sc && !test_sc && !draw_hidden)
    {
        if (draw_colors)
            currcolor = trimesh::vec(0.5, 0.5, 1.0);
        else
            currcolor = trimesh::vec(0.6, 0.6, 0.6);
        glLineWidth(1.5);
        glBegin(GL_LINES);
        draw_isolines({kr, sctest_num, sctest_den, ndotv, true, !!use_hermite,
                       false, 0.0f},
                      mesh, viewpos, currcolor);
        glEnd();
        currcolor = trimesh::vec(0.0, 0.0, 0.0);
    }

    // Suggestive contours and contours
    if (draw_sc && !use_texture)
    {
        float fade = draw_faded ? 0.03f / trimesh::sqr(feature_size) : 0.0f;
        if (draw_colors)
            currcolor = trimesh::vec(0.0, 0.0, 0.8);
        glLineWidth(2.5);
        glBegin(GL_LINES);
        draw_isolines({kr, sctest_num, sctest_den, ndotv, true, !!use_hermite,
                       true, fade},
                      mesh, viewpos, currcolor);
        glEnd();
    }
    if (draw_c && !use_texture)
    {
        if (draw_colors)
            currcolor = trimesh::vec(0.0, 0.6, 0.0);
        glLineWidth(2.5);
        glBegin(GL_LINES);
        draw_isolines(
            {ndotv, kr, std::vector<float>(), ndotv, false, false, true, 0.0f},
            mesh, viewpos, currcolor);
        glEnd();
    }
    if ((draw_sc || draw_c) && use_texture)
        draw_c_sc_texture(ndotv, kr, sctest_num, sctest_den, mesh);

    // Boundaries
    if (draw_bdy)
        draw_boundaries(false, mesh);

    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_POINT_SMOOTH);
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
}

// Signal a redraw
void need_redraw()
{
    glutPostRedisplay();
}

// Clear the screen and reset OpenGL modes to something sane
void cls()
{
    glDisable(GL_DITHER);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_NORMALIZE);
    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
    glClearColor(1, 1, 1, 0);
    if (color_style == COLOR_GRAY)
        glClearColor(0.8, 0.8, 0.8, 0);
    glClearDepth(1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

// Set up viewport and scissoring for the subwindow, and optionally draw
// a box around it (actually, just clears a rectangle one pixel bigger
// to black).  Assumes current viewport is set up for the whole window.
void set_subwindow_viewport(bool draw_box = false)
{
    GLint V[4];
    glGetIntegerv(GL_VIEWPORT, V);
    GLint x = V[0], y = V[1], w = V[2], h = V[3];
    int   boxsize = std::min(w, h) / 3;

    x += w - boxsize * 11 / 10;
    y += h - boxsize * 11 / 10;
    w = h = boxsize;

    if (draw_box)
    {
        glViewport(x - 1, y - 1, w + 2, h + 2);
        glScissor(x - 1, y - 1, w + 2, h + 2);
        glClearColor(0, 0, 0, 0);
        glEnable(GL_SCISSOR_TEST);
        glClear(GL_COLOR_BUFFER_BIT);
        glScissor(x, y, w, h);
    }

    glViewport(x, y, w, h);
}

// Draw the scene
void redraw()
{
    using namespace trimesh;

    trimesh::timestamp t = trimesh::now();
    viewpos              = inv(xf) * trimesh::point(0, 0, 0);
    GLUI_Master.auto_set_viewport();

    // If dual viewports, first draw in window using camera_alt
    if (dual_vpmode)
    {
        // Set up camera and clear the screen
        glMatrixMode(GL_PROJECTION);
        glLoadMatrixd(alt_projmatrix);
        camera_alt.setupGL(xf_alt * glut_mesh->bsphere.center,
                           glut_mesh->bsphere.r);
        glGetDoublev(GL_PROJECTION_MATRIX, alt_projmatrix);
        cls();

        // Transform and draw
        glPushMatrix();
        glMultMatrixd((double*)xf_alt);
        draw_mesh(glut_mesh);
        glPopMatrix();

        // Set viewport and draw a box for the subwindow
        set_subwindow_viewport(true);

        // Now we're ready to draw in the subwindow
    }

    camera.setupGL(xf * glut_mesh->bsphere.center, glut_mesh->bsphere.r);

    cls();

    // Transform and draw
    glPushMatrix();
    glMultMatrixd((double*)xf);
    draw_mesh(glut_mesh);
    glPopMatrix();

    glDisable(GL_SCISSOR_TEST);
    glutSwapBuffers();
    printf("\rElapsed time: %.2f msec.", 1000.0f * (trimesh::now() - t));
    fflush(stdout);

    // See if we need to autospin the camera(s)
    if (camera.autospin(xf))
        need_redraw();
    if (dual_vpmode)
    {
        if (camera_alt.autospin(xf_alt))
            need_redraw();
    }
    else
    {
        camera_alt = camera;
        xf_alt     = xf;
    }
}

// Set the view to look at the middle of the mesh, from reasonably far away
void resetview()
{
    camera.stopspin();
    camera_alt.stopspin();

    if (!xf.read(xffilename))
        xf = trimesh::xform::trans(0, 0, -3.5f / fov * glut_mesh->bsphere.r) *
             trimesh::xform::trans(-glut_mesh->bsphere.center);
    camera_alt = camera;
    xf_alt     = xf;

    // Reset light position too
    lightdir->reset();
}

// Smooth the mesh
void filter_mesh(int dummy = 0)
{
    printf("\r");
    fflush(stdout);
    smooth_mesh(glut_mesh, currsmooth);

    if (use_dlists)
    {
        glDeleteLists(1, 1);
    }
    glut_mesh->pointareas.clear();
    glut_mesh->normals.clear();
    glut_mesh->curv1.clear();
    glut_mesh->dcurv.clear();
    glut_mesh->need_normals();
    glut_mesh->need_curvatures();
    glut_mesh->need_dcurv();
    curv_colors.clear();
    gcurv_colors.clear();
    currsmooth *= 1.1f;
}

// Diffuse the normals across the mesh
void filter_normals(int dummy = 0)
{
    printf("\r");
    fflush(stdout);
    diffuse_normals(glut_mesh, currsmooth);
    glut_mesh->curv1.clear();
    glut_mesh->dcurv.clear();
    glut_mesh->need_curvatures();
    glut_mesh->need_dcurv();
    curv_colors.clear();
    gcurv_colors.clear();
    currsmooth *= 1.1f;
}

// Diffuse the curvatures across the mesh
void filter_curv(int dummy = 0)
{
    printf("\r");
    fflush(stdout);
    diffuse_curv(glut_mesh, currsmooth);
    glut_mesh->dcurv.clear();
    glut_mesh->need_dcurv();
    curv_colors.clear();
    gcurv_colors.clear();
    currsmooth *= 1.1f;
}

// Diffuse the curvature derivatives across the mesh
void filter_dcurv(int dummy = 0)
{
    printf("\r");
    fflush(stdout);
    diffuse_dcurv(glut_mesh, currsmooth);
    curv_colors.clear();
    gcurv_colors.clear();
    currsmooth *= 1.1f;
}

// Perform an iteration of subdivision
void subdivide_mesh(int dummy = 0)
{
    printf("\r");
    fflush(stdout);
    subdiv(glut_mesh);

    if (use_dlists)
    {
        glDeleteLists(1, 1);
    }
    glut_mesh->need_tstrips();
    glut_mesh->need_normals();
    glut_mesh->need_pointareas();
    glut_mesh->need_curvatures();
    glut_mesh->need_dcurv();
    curv_colors.clear();
    gcurv_colors.clear();
}

// Save the current image to a PPM file
void dump_image(int dummy = 0)
{
    // Find first non-used filename
    const char filenamepattern[] = "img%d.ppm";
    int        imgnum            = 0;
    FILE*      f;
    while (1)
    {
        char filename[1024];
        sprintf(filename, filenamepattern, imgnum++);
        f = fopen(filename, "rb");
        if (!f)
        {
            f = fopen(filename, "wb");
            printf("\n\nSaving image %s... ", filename);
            fflush(stdout);
            break;
        }
        fclose(f);
    }

    // Read pixels
    GLUI_Master.auto_set_viewport();
    GLint V[4];
    glGetIntegerv(GL_VIEWPORT, V);
    GLint width = V[2], height = V[3];
    char* buf = new char[width * height * 3];
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(V[0], V[1], width, height, GL_RGB, GL_UNSIGNED_BYTE, buf);

    // Flip top-to-bottom
    for (int i = 0; i < height / 2; i++)
    {
        char* row1 = buf + 3 * width * i;
        char* row2 = buf + 3 * width * (height - 1 - i);
        for (int j = 0; j < 3 * width; j++)
            std::swap(row1[j], row2[j]);
    }

    // Write out file
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    fwrite(buf, width * height * 3, 1, f);
    fclose(f);
    delete[] buf;

    printf("Done.\n\n");
}

// Compute a "feature size" for the mesh: computed as 1% of
// the reciprocal of the 10-th percentile curvature
void compute_feature_size()
{
    int nv    = glut_mesh->curv1.size();
    int nsamp = std::min(nv, 500);

    std::vector<float> samples;
    samples.reserve(nsamp * 2);

    for (int i = 0; i < nsamp; i++)
    {
        // Quick 'n dirty portable random number generator
        static unsigned randq = 0;
        randq = unsigned(1664525) * randq + unsigned(1013904223);

        int ind = randq % nv;
        samples.push_back(fabs(glut_mesh->curv1[ind]));
        samples.push_back(fabs(glut_mesh->curv2[ind]));
    }

    const float frac = 0.1f;
    const float mult = 0.01f;
    glut_mesh->need_bsphere();
    float max_feature_size = 0.05f * glut_mesh->bsphere.r;

    int which = int(frac * samples.size());
    std::nth_element(samples.begin(), samples.begin() + which, samples.end());

    feature_size = std::min(mult / samples[which], max_feature_size);
}

// Handle mouse button and motion events
static unsigned       buttonstate  = 0;
static const unsigned ctrl_pressed = 1 << 30;

void mousemotionfunc(int x, int y)
{
    // Ctrl+mouse = relight
    if (buttonstate & ctrl_pressed)
    {
        GLUI_Master.auto_set_viewport();
        GLint V[4];
        glGetIntegerv(GL_VIEWPORT, V);
        y        = V[1] + V[3] - 1 - y; // Adjust for top-left vs. bottom-left
        float xx = 2.0f * float(x - V[0]) / float(V[2]) - 1.0f;
        float yy = 2.0f * float(y - V[1]) / float(V[3]) - 1.0f;
        float theta = M_PI * std::min(sqrtf(xx * xx + yy * yy), 1.0f);
        float phi   = atan2(yy, xx);
        trimesh::XForm<float> lightxf =
            lightxf.rot(phi, 0, 0, 1) * lightxf.rot(theta, 0, 1, 0);
        lightdir->set_float_array_val((float*)lightxf);
        need_redraw();
        return;
    }

    static const trimesh::Mouse::button physical_to_logical_map[] = {
        trimesh::Mouse::NONE,   trimesh::Mouse::ROTATE, trimesh::Mouse::MOVEXY,
        trimesh::Mouse::MOVEZ,  trimesh::Mouse::MOVEZ,  trimesh::Mouse::MOVEXY,
        trimesh::Mouse::MOVEXY, trimesh::Mouse::MOVEXY,
    };
    trimesh::Mouse::button b = trimesh::Mouse::NONE;
    if (buttonstate & (1 << 3))
        b = trimesh::Mouse::WHEELUP;
    else if (buttonstate & (1 << 4))
        b = trimesh::Mouse::WHEELDOWN;
    else
        b = physical_to_logical_map[buttonstate & 7];

    if (dual_vpmode && mouse_moves_alt)
    {
        GLUI_Master.auto_set_viewport();
        glMatrixMode(GL_PROJECTION);
        glLoadMatrixd(alt_projmatrix);
        camera_alt.setupGL(xf_alt * glut_mesh->bsphere.center,
                           glut_mesh->bsphere.r);
        camera_alt.mouse(x, y, b, xf_alt * glut_mesh->bsphere.center,
                         glut_mesh->bsphere.r, xf_alt);
    }
    else
    {
        camera.mouse(x, y, b, xf * glut_mesh->bsphere.center, glut_mesh->bsphere.r,
                     xf);
    }

    need_redraw();
    GLUI_Master.sync_live_all();
}

void mousebuttonfunc(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        buttonstate |= (1 << button);
    else
        buttonstate &= ~(1 << button);

    if (glutGetModifiers() & GLUT_ACTIVE_CTRL)
        buttonstate |= ctrl_pressed;
    else
        buttonstate &= ~ctrl_pressed;

    // On a mouse click in dual_vpmode, check if we're in subwindow
    if (dual_vpmode && state == GLUT_DOWN && !(buttonstate & ctrl_pressed))
    {
        GLUI_Master.auto_set_viewport();
        GLint V[4];
        glGetIntegerv(GL_VIEWPORT, V);
        int yy = V[1] + V[3] - 1 - y; // top-left vs. bottom-left
        set_subwindow_viewport();
        glGetIntegerv(GL_VIEWPORT, V);
        mouse_moves_alt =
            !(x > V[0] && yy > V[1] && x < V[0] + V[2] && yy < V[1] + V[3]);
    }

    mousemotionfunc(x, y);
}

#define Ctrl (1 - 'a')

// Keyboard callback
void keyboardfunc(unsigned char key, int x, int y)
{
    switch (key)
    {
    case ' ':
        resetview();
        break;
    case 'a':
        draw_asymp = !draw_asymp;
        break;
    case 'A':
        draw_apparent = !draw_apparent;
        break;
    case 'b':
        draw_bdy = !draw_bdy;
        break;
    case 'B':
        test_c = !test_c;
        break;
    case 'c':
        draw_colors = !draw_colors;
        break;
    case 'C':
        filter_curv();
        break;
    case 'd':
        draw_c = !draw_c;
        break;
    case 'D':
        draw_sc = !draw_sc;
        break;
    case 'e':
        draw_edges = !draw_edges;
        break;
    case 'E':
        draw_extsil = !draw_extsil;
        break;
    case 'f':
        draw_faded = !draw_faded;
        break;
    case 'F':
        draw_sh = !draw_sh;
        break;
    case 'g':
        use_hermite = !use_hermite;
        break;
    case 'h':
        draw_hidden = !draw_hidden;
        break;
    case 'H':
        draw_H = !draw_H;
        break;
    case 'i':
        draw_isoph = !draw_isoph;
        break;
    case 'I':
        dump_image();
        break;
    case 'K':
        draw_K = !draw_K;
        break;
    case 'l':
        lighting_style++;
        lighting_style %= nlighting_styles;
        break;
    case 'n':
        draw_norm = !draw_norm;
        break;
    case 'r':
        draw_phridges = !draw_phridges;
        break;
    case Ctrl + 'r':
        draw_ridges = !draw_ridges;
        break;
    case 'R':
        test_rv = !test_rv;
        break;
    case 's':
        filter_normals();
        break;
    case 'S':
        filter_mesh();
        break;
    case 't':
        use_texture = !use_texture;
        break;
    case 'T':
        test_sc = !test_sc;
        break;
    case 'u':
        color_style++;
        color_style %= ncolor_styles;
        if (color_style == COLOR_MESH && glut_mesh->colors.empty())
            color_style = COLOR_WHITE;
        break;
    case 'v':
        draw_phvalleys = !draw_phvalleys;
        break;
    case Ctrl + 'v':
        draw_valleys = !draw_valleys;
        break;
    case 'V':
    case Ctrl + 's':
        subdivide_mesh();
        break;
    case 'w':
        draw_w = !draw_w;
        break;
    case Ctrl + 'w':
        draw_wperp = !draw_wperp;
        break;
    case 'W':
    case Ctrl + 'd':
        draw_DwKr = !draw_DwKr;
        break;
    case 'x':
        xf.write(xffilename);
        break;
    case 'X':
        filter_dcurv();
        break;
    case 'z':
        fov /= 1.1f;
        camera.set_fov(fov);
        break;
    case 'Z':
        fov *= 1.1f;
        camera.set_fov(fov);
        break;
    case '/':
        dual_vpmode = !dual_vpmode;
        break;
    case '+':
    case '=':
        niso++;
        break;
    case '-':
    case '_':
        if (niso > 1)
            niso--;
        break;
    case '1':
        draw_curv1 = !draw_curv1;
        break;
    case '2':
        draw_curv2 = !draw_curv2;
        break;
    case '7':
        rv_thresh /= 1.1f;
        break;
    case '8':
        rv_thresh *= 1.1f;
        break;
    case '9':
        sug_thresh /= 1.1f;
        break;
    case '0':
        sug_thresh *= 1.1f;
        break;
    case '\033': // Esc
    case '\021': // Ctrl-Q
    case 'Q':
    case 'q':
        exit(0);
    }
    need_redraw();
    GLUI_Master.sync_live_all();
}

void skeyboardfunc(int key, int x, int y)
{
    switch (key)
    {
    case GLUT_KEY_UP:
        sug_thresh *= 1.1f;
        break;
    case GLUT_KEY_DOWN:
        sug_thresh /= 1.1f;
        break;

    case GLUT_KEY_RIGHT:
        rv_thresh *= 1.1f;
        break;
    case GLUT_KEY_LEFT:
        rv_thresh /= 1.1f;
        break;
    }
    need_redraw();
    GLUI_Master.sync_live_all();
}

// Reshape the window.  We clear the window here to possibly avoid some
// weird problems.  Yuck.
void reshape(int x, int y)
{
    GLUI_Master.auto_set_viewport();
    cls();
    glutSwapBuffers();
    need_redraw();
}

void usage(const char* myname)
{
    fprintf(stderr, "Usage: %s [-options] infile\n", myname);
    exit(1);
}

int main(int argc, char* argv[])
{
    int wwid = 820, wht = 700;
    for (int j = 1; j < argc; j++)
    {
        if (argv[j][0] == '+')
        {
            sscanf(argv[j] + 1, "%d,%d,%f,%f", &wwid, &wht, &sug_thresh,
                   &ph_thresh);
        }
    }

    glutInitWindowSize(wwid, wht);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInit(&argc, argv);

    if (argc < 2)
        usage(argv[0]);

    // Skip over any parameter beginning with a '-' or '+'
    int i = 1;
    while (i < argc - 1 && argv[i][0] == '-' && argv[i][0] == '+')
    {
        i++;
        if (!strcmp(argv[i - 1], "--"))
            break;
    }
    std::string filename{argv[i]};

    glut_mesh = trimesh::TriMesh::read(filename);
    if (!glut_mesh)
        usage(argv[0]);

    xffilename = filename.substr(0, filename.find_last_of('.')) + ".xf";

    glut_mesh->need_tstrips();
    glut_mesh->need_bsphere();
    glut_mesh->need_normals();
    glut_mesh->need_curvatures();
    glut_mesh->need_dcurv();
    compute_feature_size();
    currsmooth = 0.5f * glut_mesh->feature_size();

    char windowname[255];
    sprintf(windowname, "RTSC - %s", filename.c_str());
    int main_win = glutCreateWindow(windowname);

    glutDisplayFunc(redraw);
    GLUI_Master.set_glutMouseFunc(mousebuttonfunc);
    glutMotionFunc(mousemotionfunc);
    GLUI_Master.set_glutKeyboardFunc(keyboardfunc);
    GLUI_Master.set_glutSpecialFunc(skeyboardfunc);
    GLUI_Master.set_glutReshapeFunc(reshape);

    GLUI* glui =
        GLUI_Master.create_glui_subwindow(main_win, GLUI_SUBWINDOW_BOTTOM);
    glui->set_main_gfx_window(main_win);
    GLUI_Rollout* g = glui->add_rollout("Options", false);
    glui->add_statictext_to_panel(g, "Lines:");
    glui->add_checkbox_to_panel(g, "Exterior silhouette", &draw_extsil);
    glui->add_checkbox_to_panel(g, "Occluding contours", &draw_c);
    glui->add_checkbox_to_panel(g, "Suggestive contours", &draw_sc);

    glui->add_checkbox_to_panel(g, "Suggestive hlt.", &draw_sh);
    glui->add_checkbox_to_panel(g, "Principal hlt. (R)", &draw_phridges);
    glui->add_checkbox_to_panel(g, "Principal hlt. (V)", &draw_phvalleys);

    glui->add_checkbox_to_panel(g, "Ridges", &draw_ridges);
    glui->add_checkbox_to_panel(g, "Valleys", &draw_valleys);

    glui->add_checkbox_to_panel(g, "Apparent ridges", &draw_apparent);

    glui->add_checkbox_to_panel(g, "K = 0", &draw_K);
    glui->add_checkbox_to_panel(g, "H = 0", &draw_H);
    glui->add_checkbox_to_panel(g, "DwKr = thresh.", &draw_DwKr);

    glui->add_checkbox_to_panel(g, "Boundaries", &draw_bdy);

    glui->add_checkbox_to_panel(g, "Isophotes", &draw_isoph);
    GLUI_Spinner* spinner =
        glui->add_spinner_to_panel(g, "# Isophotes ", GLUI_SPINNER_INT, &niso);
    spinner->set_int_limits(1, 100);
    spinner->edittext->set_w(120);

    glui->add_checkbox_to_panel(g, "Topo lines", &draw_topo);
    spinner =
        glui->add_spinner_to_panel(g, "# Topo lines", GLUI_SPINNER_INT, &ntopo);
    spinner->set_int_limits(1, 100);
    spinner->edittext->set_w(120);
    glui->add_slider_to_panel(g, "Topo offset", GLUI_SLIDER_FLOAT, -1.0, 1.0,
                              &topo_offset)
        ->set_w(5);

    glui->add_column_to_panel(g, false);
    glui->add_statictext_to_panel(g, "Line tests:");
    glui->add_checkbox_to_panel(g, "Draw hidden lines", &draw_hidden);
    glui->add_checkbox_to_panel(g, "Trim \"inside\" contours", &test_c);
    glui->add_checkbox_to_panel(g, "Trim SC", &test_sc);
    glui->add_slider_to_panel(g, "SC thresh", GLUI_SLIDER_FLOAT, 0.0, 0.1,
                              &sug_thresh);
    glui->add_checkbox_to_panel(g, "Trim SH", &test_sh);
    glui->add_slider_to_panel(g, "SH thresh", GLUI_SLIDER_FLOAT, 0.0, 0.1,
                              &sh_thresh);
    glui->add_checkbox_to_panel(g, "Trim PH", &test_ph);
    glui->add_slider_to_panel(g, "PH thresh", GLUI_SLIDER_FLOAT, 0.0, 0.2,
                              &ph_thresh);
    glui->add_checkbox_to_panel(g, "Trim RV", &test_rv);
    glui->add_slider_to_panel(g, "RV thresh", GLUI_SLIDER_FLOAT, 0.0, 0.5,
                              &rv_thresh);
    glui->add_checkbox_to_panel(g, "Trim AR", &test_ar);
    glui->add_slider_to_panel(g, "AR thresh", GLUI_SLIDER_FLOAT, 0.0, 0.5,
                              &ar_thresh);

    glui->add_column_to_panel(g, false);
    glui->add_statictext_to_panel(g, "Line style:");
    glui->add_checkbox_to_panel(g, "Texture mapping", &use_texture);
    glui->add_checkbox_to_panel(g, "Fade lines", &draw_faded);
    glui->add_checkbox_to_panel(g, "Draw in color", &draw_colors);
    glui->add_checkbox_to_panel(g, "Hermite interp", &use_hermite);

    glui->add_statictext_to_panel(g, " ");
    glui->add_statictext_to_panel(g, "Mesh style:");
    GLUI_RadioGroup* r = glui->add_radiogroup_to_panel(g, &color_style);
    glui->add_radiobutton_to_group(r, "White");
    glui->add_radiobutton_to_group(r, "Gray");
    glui->add_radiobutton_to_group(r, "Curvature (color)");
    glui->add_radiobutton_to_group(r, "Curvature (gray)");
    if (!glut_mesh->colors.empty())
        glui->add_radiobutton_to_group(r, "Mesh colors");
    glui->add_checkbox_to_panel(g, "Draw edges", &draw_edges);

    glui->add_column_to_panel(g, false);
    glui->add_statictext_to_panel(g, "Lighting:");
    r = glui->add_radiogroup_to_panel(g, &lighting_style);
    glui->add_radiobutton_to_group(r, "None");
    glui->add_radiobutton_to_group(r, "Lambertian");
    glui->add_radiobutton_to_group(r, "Lambertian2");
    glui->add_radiobutton_to_group(r, "Hemisphere");
    glui->add_radiobutton_to_group(r, "Toon (gray/white)");
    glui->add_radiobutton_to_group(r, "Toon (black/white)");
    glui->add_radiobutton_to_group(r, "Gooch");
    lightdir =
        glui->add_rotation_to_panel(g, "Direction", (float*)&lightdir_matrix);
    glui->add_checkbox_to_panel(g, "On camera", &light_wrt_camera);
    lightdir->reset();

    glui->add_column_to_panel(g, false);
    glui->add_statictext_to_panel(g, "Vectors:");
    glui->add_checkbox_to_panel(g, "Normal", &draw_norm);
    glui->add_checkbox_to_panel(g, "Principal 1", &draw_curv1);
    glui->add_checkbox_to_panel(g, "Principal 2", &draw_curv2);
    glui->add_checkbox_to_panel(g, "Asymptotic", &draw_asymp);
    glui->add_checkbox_to_panel(g, "Proj. View", &draw_w);

    glui->add_statictext_to_panel(g, " ");
    glui->add_statictext_to_panel(g, "Camera:");
    glui->add_checkbox_to_panel(g, "Dual viewport", &dual_vpmode);

    glui->add_column_to_panel(g, false);
    glui->add_button_to_panel(g, "Smooth Mesh", 0, filter_mesh);
    glui->add_button_to_panel(g, "Smooth Normals", 0, filter_normals);
    glui->add_button_to_panel(g, "Smooth Curv", 0, filter_curv);
    glui->add_button_to_panel(g, "Smooth DCurv", 0, filter_dcurv);
    glui->add_button_to_panel(g, "Subdivide Mesh", 0, subdivide_mesh);
    glui->add_button_to_panel(g, "Screencap", 0, dump_image);
    glui->add_button_to_panel(g, "Exit", 0, exit);

    // Go through command-line arguments and do what they say.
    // Any command line options are just interpreted as keyboard commands.
    for (i = 1; i < argc - 1; i++)
    {
        if (argv[i][0] != '-' || !strcmp(argv[i], "--"))
            break;
        for (size_t j = 1; j < strlen(argv[i]); j++)
            keyboardfunc(argv[i][j], 0, 0);
    }

    resetview();

    glutMainLoop();
}

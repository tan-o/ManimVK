// A C interface to Intel's Masked Occlusion Culling for ManimVK's Rust code.
#include <cstddef>

#include "MaskedOcclusionCulling.h"

extern "C" {

void *moc_create() { return MaskedOcclusionCulling::Create(); }

void moc_destroy(void *moc) {
    MaskedOcclusionCulling::Destroy(static_cast<MaskedOcclusionCulling *>(moc));
}

void moc_set_resolution(void *moc, unsigned int width, unsigned int height) {
    static_cast<MaskedOcclusionCulling *>(moc)->SetResolution(width, height);
}

void moc_set_near_clip_plane(void *moc, float near_distance) {
    static_cast<MaskedOcclusionCulling *>(moc)->SetNearClipPlane(near_distance);
}

void moc_clear(void *moc) { static_cast<MaskedOcclusionCulling *>(moc)->ClearBuffer(); }

// Vertices are (x, y, w) clip coordinates, three floats each; no back faces
// are culled.
void moc_render_triangles(void *moc, const float *vertices, const unsigned int *triangles,
                          int count) {
    static_cast<MaskedOcclusionCulling *>(moc)->RenderTriangles(
        vertices, triangles, count, nullptr, MaskedOcclusionCulling::BACKFACE_NONE,
        MaskedOcclusionCulling::CLIP_PLANE_ALL, MaskedOcclusionCulling::VertexLayout(12, 4, 8));
}

// 0 visible, 1 occluded, 3 outside the view.
int moc_test_rect(const void *moc, float xmin, float ymin, float xmax, float ymax, float wmin) {
    return static_cast<const MaskedOcclusionCulling *>(moc)->TestRect(xmin, ymin, xmax, ymax,
                                                                       wmin);
}
}

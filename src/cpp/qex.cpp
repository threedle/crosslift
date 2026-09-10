// libQEx: seamless UVs -> quad mesh.

#include "crosslift/quadrangulate.h"

#include <qex.h>

#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace crosslift {
namespace {

/// QEx allocates with malloc and expects the caller to free, so RAII it.
struct MallocDeleter {
    void operator()(void *p) const noexcept { std::free(p); }
};
template <typename T>
using malloc_ptr = std::unique_ptr<T, MallocDeleter>;

template <typename T>
malloc_ptr<T> malloc_array(std::size_t n) {
    auto *p = static_cast<T *>(std::malloc(sizeof(T) * n));
    if (p == nullptr) throw std::bad_alloc();
    return malloc_ptr<T>(p);
}

}  // namespace

void qex_extract(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                 const Eigen::MatrixXd &UV, const Eigen::MatrixXi &FUV,
                 Eigen::MatrixXd &quad_V, Eigen::MatrixXi &quad_F) {
    const auto nV = static_cast<std::size_t>(V.rows());
    const auto nF = static_cast<std::size_t>(F.rows());

    auto vertices = malloc_array<qex_Point3>(nV);
    auto tris = malloc_array<qex_Tri>(nF);
    auto uv_tris = malloc_array<qex_UVTri>(nF);

    for (std::size_t i = 0; i < nV; ++i)
        for (int j = 0; j < 3; ++j) vertices.get()[i].x[j] = V(static_cast<int>(i), j);

    for (std::size_t i = 0; i < nF; ++i)
        for (int j = 0; j < 3; ++j)
            tris.get()[i].indices[j] = static_cast<qex_Index>(F(static_cast<int>(i), j));

    // QEx wants UVs per triangle corner, not indexed — expand through FUV.
    for (std::size_t i = 0; i < nF; ++i) {
        for (int j = 0; j < 3; ++j) {
            const int uv_idx = FUV(static_cast<int>(i), j);
            uv_tris.get()[i].uvs[j].x[0] = UV(uv_idx, 0);
            uv_tris.get()[i].uvs[j].x[1] = UV(uv_idx, 1);
        }
    }

    qex_TriMesh tri{};
    tri.vertex_count = static_cast<unsigned int>(nV);
    tri.tri_count = static_cast<unsigned int>(nF);
    tri.vertices = vertices.get();
    tri.tris = tris.get();
    tri.uvTris = uv_tris.get();

    qex_QuadMesh quad{};
    qex_extractQuadMesh(&tri, nullptr, &quad);

    // Adopt QEx's output so it is freed even if the copies below throw.
    malloc_ptr<qex_Point3> quad_vertices(quad.vertices);
    malloc_ptr<qex_Quad> quad_quads(quad.quads);

    quad_V.resize(quad.vertex_count, 3);
    for (unsigned int i = 0; i < quad.vertex_count; ++i)
        for (int j = 0; j < 3; ++j) quad_V(i, j) = quad.vertices[i].x[j];

    quad_F.resize(quad.quad_count, 4);
    for (unsigned int i = 0; i < quad.quad_count; ++i)
        for (int j = 0; j < 4; ++j)
            quad_F(i, j) = static_cast<int>(quad.quads[i].indices[j]);
}

}  // namespace crosslift

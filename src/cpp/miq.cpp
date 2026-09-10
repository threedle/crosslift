// MIQ parameterization and the curvature heuristic that sizes it.

#include "crosslift/quadrangulate.h"

#include <igl/copyleft/comiso/miq.h>
#include <igl/doublearea.h>
#include <igl/edge_lengths.h>
#include <igl/per_face_normals.h>
#include <igl/triangle_triangle_adjacency.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace crosslift {

double adaptive_gradient_size(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                              double scale_factor) {
    Eigen::VectorXd dblA;
    igl::doublearea(V, F, dblA);
    const double total_area = dblA.sum() / 2.0;

    Eigen::MatrixXd el;
    igl::edge_lengths(V, F, el);

    Eigen::MatrixXd N;
    igl::per_face_normals(V, F, N);

    Eigen::MatrixXi TT, TTi;
    igl::triangle_triangle_adjacency(F, TT, TTi);

    // Total absolute curvature: dihedral angle weighted by the edge it turns
    // across. Each interior edge is visited once (`f >= adj_f` skips the twin).
    double curvature_integral = 0.0;
    for (int f = 0; f < F.rows(); ++f) {
        for (int e = 0; e < 3; ++e) {
            const int adj_f = TT(f, e);
            if (adj_f == -1 || f >= adj_f) continue;
            const Eigen::Vector3d n1 = N.row(f);
            const Eigen::Vector3d n2 = N.row(adj_f);
            const double cos_angle = std::max(-1.0, std::min(1.0, n1.dot(n2)));
            curvature_integral += el(f, e) * std::acos(cos_angle);
        }
    }

    // A plane has nothing to measure; fall back to a pure area heuristic.
    constexpr double min_curvature = 1e-8;
    const double sqrt_area = std::sqrt(total_area);
    if (curvature_integral < min_curvature) return scale_factor * sqrt_area;

    return scale_factor * std::sqrt(curvature_integral / sqrt_area);
}

void miq_parameterize(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                      Eigen::MatrixXd PD1, Eigen::MatrixXd PD2,
                      double gradient_size, double stiffness, bool direct_round,
                      const Eigen::VectorXd *h_per_face, bool normalize_frames,
                      Eigen::MatrixXd &UV, Eigen::MatrixXi &FUV) {
    Eigen::MatrixXd N_faces;
    igl::per_face_normals(V, F, N_faces);

    if (normalize_frames) {
        // MIQ reads frame lengths as gradient scaling; start unit-orthogonal
        // so gradient_size (plus h_per_face below) is the only density knob.
        for (int i = 0; i < F.rows(); ++i) {
            if (PD1.row(i).norm() > 1e-12) {
                PD1.row(i).normalize();
                const Eigen::Vector3d n = N_faces.row(i);
                const Eigen::Vector3d p = PD1.row(i);
                PD2.row(i) = n.cross(p).normalized();
            }
        }
    } else {
        // Pass frames through untouched so their lengths reach the Poisson RHS;
        // only degenerate (zero-length) rows are repaired.
        int repaired = 0;
        for (int i = 0; i < F.rows(); ++i) {
            if (PD1.row(i).norm() > 1e-12 && PD2.row(i).norm() > 1e-12) continue;
            const Eigen::Vector3d n = N_faces.row(i);
            const Eigen::Vector3d a =
                (std::abs(n.x()) < 0.9) ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
            const Eigen::Vector3d p = n.cross(a).normalized();
            PD1.row(i) = p;
            PD2.row(i) = n.cross(p).normalized();
            ++repaired;
        }
        std::cout << "[miq] frames passed through unnormalized (" << repaired
                  << " degenerate frame(s) repaired)" << std::endl;
    }

    if (h_per_face != nullptr) {
        if (h_per_face->size() != F.rows())
            throw std::runtime_error("h_per_face length must equal the number of faces");
        // Longer frames -> faster UV change -> smaller quads, so scale by 1/h.
        int scaled = 0;
        for (int i = 0; i < F.rows(); ++i) {
            const double hi = (*h_per_face)(i);
            if (hi > 1e-12 && PD1.row(i).norm() > 1e-12) {
                PD1.row(i) /= hi;
                PD2.row(i) /= hi;
                ++scaled;
            }
        }
        std::cout << "[miq] per-face sizing applied to " << scaled << "/" << F.rows()
                  << " faces" << std::endl;
    }

    igl::copyleft::comiso::miq(V, F, PD1, PD2, UV, FUV, gradient_size, stiffness,
                               direct_round,
                               5,     // stiffness iterations
                               5,     // local integer rounding iterations
                               true,  // doRound
                               true   // singularityRound
    );
}

}  // namespace crosslift

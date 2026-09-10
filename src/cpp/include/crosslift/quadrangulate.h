// Public C++ API for the field -> quad mesh stage.

#pragma once

#include <Eigen/Core>

namespace crosslift {

/// Pick MIQ's global gradient size from how curved the mesh is.
///
/// Integrates dihedral angle over edge length, normalizes by sqrt(area) so the
/// result is scale-invariant, and returns `scale_factor * sqrt(that)`. A flat
/// mesh has no curvature to measure, so it falls back to `scale_factor *
/// sqrt(area)`.
double adaptive_gradient_size(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                              double scale_factor);

/// Mixed-integer quadrangulation (libigl / CoMISo): cross field -> seamless UVs.
///
/// @param PD1,PD2  per-face frame, taken by value because it is conditioned in
///                 place before the solve.
/// @param h_per_face  optional (F,) relative target edge length; frames are
///                 scaled by 1/h(f), so larger h means larger quads. Pass
///                 nullptr for uniform sizing. Give it mean ~1 to keep
///                 `gradient_size` the global knob.
/// @param normalize_frames  when true, PD1 is unitized and PD2 rebuilt
///                 orthogonal to it, so only the field's *direction* reaches
///                 the solver. Set false to let frame lengths through — that is
///                 what makes an anisotropic field drive per-axis quad density.
void miq_parameterize(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                      Eigen::MatrixXd PD1, Eigen::MatrixXd PD2,
                      double gradient_size, double stiffness, bool direct_round,
                      const Eigen::VectorXd *h_per_face, bool normalize_frames,
                      Eigen::MatrixXd &UV, Eigen::MatrixXi &FUV);

/// libQEx: seamless UVs -> quad mesh.
void qex_extract(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                 const Eigen::MatrixXd &UV, const Eigen::MatrixXi &FUV,
                 Eigen::MatrixXd &quad_V, Eigen::MatrixXi &quad_F);

/// Directional's seamless integration: N-RoSy field -> N parametric functions.
///
/// The N != 4 counterpart to MIQ, and the candidate replacement for it at N = 4.
/// Runs principal matching, cuts the mesh along the seams, and solves the
/// Poisson system.
///
/// @param ext_field  (F, 3N) raw extrinsic field, direction k in columns 3k..3k+2.
/// @throws std::runtime_error if the integration solver fails.
void nrosy_integrate(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                     const Eigen::MatrixXd &ext_field, int N, double length_ratio,
                     bool integral_seamless, bool round_seams,
                     Eigen::MatrixXd &n_function, Eigen::MatrixXd &n_corner_functions,
                     Eigen::MatrixXd &cut_V, Eigen::MatrixXi &cut_F);

/// Trace streamlines through an N-directional field, for figures.
///
/// @param seed_faces  face indices to seed from. When non-empty this skips
///                 Directional's own poisson-disk sampling pass, whose cost
///                 grows quadratically in `dist_ratio`.
/// @param nodes    (2S, 3) segment endpoints, @param edges (S, 2) index pairs
///                 into them, @param colors (S,) which of the N directions each
///                 segment follows.
void trace_streamlines(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                       const Eigen::MatrixXd &ext_field, int N, int num_steps,
                       double dist_ratio, const Eigen::VectorXi &seed_faces,
                       Eigen::MatrixXd &nodes, Eigen::MatrixXi &edges,
                       Eigen::VectorXi &colors);

}  // namespace crosslift

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include <optional>
#include <tuple>

#include "crosslift/quadrangulate.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_core, m) {
    m.doc() =
        "crosslift C++ core: MIQ parameterization, quad extraction, seamless "
        "integration and streamline tracing.";

    m.def("adaptive_gradient_size", &crosslift::adaptive_gradient_size, "V"_a, "F"_a,
          "scale_factor"_a = 16.0,
          "Pick MIQ's gradient size from mesh curvature.\n\n"
          "Args:\n"
          "    V: (n, 3) float64 vertex positions\n"
          "    F: (m, 3) int32 face indices\n"
          "    scale_factor: multiplier (default 16.0)\n\n"
          "Returns:\n"
          "    gradient_size (float)");

    m.def(
        "miq_parameterize",
        [](const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::MatrixXd &PD1,
           const Eigen::MatrixXd &PD2, double gradient_size, double stiffness,
           bool direct_round, std::optional<Eigen::VectorXd> h_per_face,
           bool normalize_frames) {
            Eigen::MatrixXd UV;
            Eigen::MatrixXi FUV;
            crosslift::miq_parameterize(V, F, PD1, PD2, gradient_size, stiffness,
                                        direct_round,
                                        h_per_face ? &*h_per_face : nullptr,
                                        normalize_frames, UV, FUV);
            return std::make_tuple(std::move(UV), std::move(FUV));
        },
        "V"_a, "F"_a, "PD1"_a, "PD2"_a, "gradient_size"_a = 80.0, "stiffness"_a = 5.0,
        "direct_round"_a = false, "h_per_face"_a = nb::none(), "normalize_frames"_a = true,
        "Run MIQ (Mixed-Integer Quadrangulation) parameterization.\n\n"
        "Args:\n"
        "    V: (n, 3) float64 vertex positions\n"
        "    F: (m, 3) int32 face indices\n"
        "    PD1: (m, 3) float64 first frame direction per face\n"
        "    PD2: (m, 3) float64 second frame direction per face\n"
        "    gradient_size: global quad resolution multiplier (default 80.0)\n"
        "    stiffness: MIQ stiffness weight (default 5.0)\n"
        "    direct_round: use greedy rounding instead of mixed-integer (default False)\n"
        "    h_per_face: (m,) float64 optional relative target edge length. Unit\n"
        "        frames are scaled by 1/h(f), so larger h means larger quads.\n"
        "        Give it mean ~1 so gradient_size stays the global knob.\n"
        "    normalize_frames: when True (default) PD1 is unitized and PD2 rebuilt\n"
        "        orthogonal to it, discarding frame lengths. Set False to pass\n"
        "        lengths straight to MIQ's Poisson RHS, so an anisotropic field\n"
        "        drives per-axis quad density.\n\n"
        "Returns:\n"
        "    (UV, FUV) where UV is (p, 2) float64 and FUV is (m, 3) int32");

    m.def(
        "qex_extract",
        [](const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::MatrixXd &UV,
           const Eigen::MatrixXi &FUV) {
            Eigen::MatrixXd quad_V;
            Eigen::MatrixXi quad_F;
            crosslift::qex_extract(V, F, UV, FUV, quad_V, quad_F);
            return std::make_tuple(std::move(quad_V), std::move(quad_F));
        },
        "V"_a, "F"_a, "UV"_a, "FUV"_a,
        "Extract a quad mesh from a parameterized triangle mesh using libQEx.\n\n"
        "Args:\n"
        "    V: (n, 3) float64 vertex positions\n"
        "    F: (m, 3) int32 face indices\n"
        "    UV: (p, 2) float64 UV coordinates\n"
        "    FUV: (m, 3) int32 face-to-UV indices\n\n"
        "Returns:\n"
        "    (quad_V, quad_F) where quad_V is (q, 3) float64 and quad_F is (r, 4) int32");

    m.def(
        "nrosy_integrate",
        [](const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
           const Eigen::MatrixXd &ext_field, int N, double length_ratio,
           bool integral_seamless, bool round_seams) {
            Eigen::MatrixXd n_function, n_corner_functions, cut_V;
            Eigen::MatrixXi cut_F;
            crosslift::nrosy_integrate(V, F, ext_field, N, length_ratio, integral_seamless,
                                       round_seams, n_function, n_corner_functions, cut_V,
                                       cut_F);
            return std::make_tuple(std::move(n_function), std::move(n_corner_functions),
                                   std::move(cut_V), std::move(cut_F));
        },
        "V"_a, "F"_a, "ext_field"_a, "N"_a, "length_ratio"_a = 0.02,
        "integral_seamless"_a = false, "round_seams"_a = true,
        "Integrate an N-RoSy field into N parametric functions (Directional).\n\n"
        "Principal matching, cut-mesh construction and Poisson integration, for a\n"
        "seamless parameterization from an arbitrary N-directional field.\n\n"
        "Args:\n"
        "    V: (n, 3) float64 vertex positions\n"
        "    F: (m, 3) int32 face indices\n"
        "    ext_field: (m, 3*N) float64 raw extrinsic field per face\n"
        "    N: number of directions (e.g. 2, 4, 6)\n"
        "    length_ratio: parameterization density (default 0.02)\n"
        "    integral_seamless: enforce integer seamless transitions (default False)\n"
        "    round_seams: round seam transitions rather than singularities (default True)\n\n"
        "Returns:\n"
        "    (NFunction, NCornerFunctions, cutV, cutF) where\n"
        "    NFunction is (cV, N) float64 per cut-vertex parametric values,\n"
        "    NCornerFunctions is (m, 3*N) float64 per-corner parametric values,\n"
        "    cutV is (cV, 3) float64 and cutF is (m, 3) int32");

    m.def(
        "trace_streamlines",
        [](const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
           const Eigen::MatrixXd &ext_field, int N, int num_steps, double dist_ratio,
           std::optional<Eigen::VectorXi> seed_faces) {
            Eigen::MatrixXd nodes;
            Eigen::MatrixXi edges;
            Eigen::VectorXi colors;
            crosslift::trace_streamlines(V, F, ext_field, N, num_steps, dist_ratio,
                                         seed_faces ? *seed_faces : Eigen::VectorXi(),
                                         nodes, edges, colors);
            return std::make_tuple(std::move(nodes), std::move(edges), std::move(colors));
        },
        "V"_a, "F"_a, "ext_field"_a, "N"_a, "num_steps"_a = 50, "dist_ratio"_a = 2.0,
        "seed_faces"_a = nb::none(),
        "Trace streamlines on a surface from an N-directional tangent field.\n\n"
        "Args:\n"
        "    V: (n, 3) float64 vertex positions\n"
        "    F: (m, 3) int32 face indices\n"
        "    ext_field: (m, 3*N) float64 raw extrinsic field per face\n"
        "    N: number of directions (e.g. 4 for a cross field)\n"
        "    num_steps: integration steps (default 50)\n"
        "    dist_ratio: seed spacing ratio (default 2.0). Ignored when seed_faces is set.\n"
        "    seed_faces: optional (k,) int32 face indices to seed from. Bypasses the\n"
        "        internal poisson-disk seeding, whose cost grows quadratically in\n"
        "        dist_ratio.\n\n"
        "Returns:\n"
        "    (nodes, edges, colors) where nodes is (2s, 3) float64, edges is (s, 2)\n"
        "    int32, and colors is (s,) int32 direction index per segment");
}

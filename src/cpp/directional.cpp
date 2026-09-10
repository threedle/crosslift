// Directional-backed operations: seamless N-RoSy integration and streamline tracing.

#include "crosslift/quadrangulate.h"

#include <directional/CartesianField.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/TriMesh.h>
#include <directional/integrate.h>
#include <directional/principal_matching.h>
#include <directional/setup_integration.h>
#include <directional/streamlines.h>

#include <stdexcept>

namespace crosslift {
namespace {

/// Directional's setup is identical for both entry points below.
struct FieldContext {
    directional::TriMesh mesh;
    directional::PCFaceTangentBundle ftb;
    directional::CartesianField field;

    FieldContext(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                 const Eigen::MatrixXd &ext_field, int N) {
        mesh.set_mesh(V, F);
        ftb.init(mesh);
        field.init(ftb, directional::fieldTypeEnum::RAW_FIELD, N);
        field.set_extrinsic_field(ext_field);
    }
};

}  // namespace

void nrosy_integrate(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                     const Eigen::MatrixXd &ext_field, int N, double length_ratio,
                     bool integral_seamless, bool round_seams,
                     Eigen::MatrixXd &n_function, Eigen::MatrixXd &n_corner_functions,
                     Eigen::MatrixXd &cut_V, Eigen::MatrixXi &cut_F) {
    FieldContext ctx(V, F, ext_field, N);

    // Matching fixes which direction of face i continues into face j,
    // defining the singularities and hence the seams.
    directional::principal_matching(ctx.field);

    directional::IntegrationData int_data(N);
    int_data.lengthRatio = length_ratio;
    int_data.integralSeamless = integral_seamless;
    int_data.roundSeams = round_seams;

    directional::TriMesh mesh_cut;
    directional::CartesianField combed_field;
    directional::setup_integration(ctx.field, int_data, mesh_cut, combed_field);

    if (!directional::integrate(combed_field, int_data, mesh_cut, n_function,
                                n_corner_functions))
        throw std::runtime_error("nrosy_integrate: integration solver failed");

    cut_V = mesh_cut.V;
    cut_F = mesh_cut.F;
}

void trace_streamlines(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                       const Eigen::MatrixXd &ext_field, int N, int num_steps,
                       double dist_ratio, const Eigen::VectorXi &seed_faces,
                       Eigen::MatrixXd &nodes, Eigen::MatrixXi &edges,
                       Eigen::VectorXi &colors) {
    FieldContext ctx(V, F, ext_field, N);

    directional::StreamlineData sl_data;
    directional::StreamlineState sl_state;
    directional::streamlines_init(ctx.field, seed_faces, dist_ratio, sl_data, sl_state);

    // Step size is set relative to the mesh so `num_steps` means the same thing
    // across meshes of different scale.
    const double d_time = 0.5 * ctx.ftb.avgAdjLength;
    for (int i = 0; i < num_steps; ++i)
        directional::streamlines_next(sl_data, sl_state, d_time);

    const int num_segments = static_cast<int>(sl_state.segStart.size());
    nodes.resize(num_segments * 2, 3);
    edges.resize(num_segments, 2);
    colors.resize(num_segments);

    // Each segment is its own two-node edge, since streamlines die at different steps.
    for (int i = 0; i < num_segments; ++i) {
        nodes.row(2 * i) = sl_state.segStart[i];
        nodes.row(2 * i + 1) = sl_state.segEnd[i];
        edges(i, 0) = 2 * i;
        edges(i, 1) = 2 * i + 1;
        colors(i) = sl_state.segOrigVector[i];
    }
}

}  // namespace crosslift

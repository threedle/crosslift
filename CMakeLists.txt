cmake_minimum_required(VERSION 3.16)
project(crosslift_core LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Everything ends up inside one shared Python module, so all of it needs -fPIC.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CoMISo wants BLAS/LAPACK. When the build runs inside a conda environment that
# ships OpenBLAS, prefer that one — CMake will not look inside an activated
# environment on its own. Guarded on the library actually being there: an
# activated environment without OpenBLAS is common (CONDA_PREFIX is exported by
# the base environment whether or not anything numeric is installed), and
# pointing the linker at a file that does not exist turns a working system BLAS
# into a link failure.
if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/lib/libopenblas.so")
  list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
  set(BLAS_LIBRARIES   "$ENV{CONDA_PREFIX}/lib/libopenblas.so" CACHE FILEPATH "" FORCE)
  set(LAPACK_LIBRARIES "$ENV{CONDA_PREFIX}/lib/libopenblas.so" CACHE FILEPATH "" FORCE)
endif()

# ============================================================================
# Python + nanobind
# ============================================================================
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

include(FetchContent)

# Cache downloads in the source tree, not the wheel build dir: scikit-build-core
# may use a fresh build directory per invocation, and re-cloning libigl + CoMISo
# + OpenMesh on every rebuild is minutes of nothing useful. Point
# CROSSLIFT_DEPS_DIR at an existing cache to share one across checkouts.
set(CROSSLIFT_DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build/_deps"
    CACHE PATH "Directory holding fetched third-party sources")
set(FETCHCONTENT_BASE_DIR "${CROSSLIFT_DEPS_DIR}" CACHE PATH "" FORCE)

# ============================================================================
# libigl v2.4.0 — for CoMISo/MIQ. Brings Eigen and CoMISo with it.
# ============================================================================
set(LIBIGL_COPYLEFT_COMISO        ON  CACHE BOOL "" FORCE)
set(LIBIGL_COPYLEFT_CORE          ON  CACHE BOOL "" FORCE)
set(LIBIGL_USE_STATIC_LIBRARY     OFF CACHE BOOL "" FORCE)
set(LIBIGL_BUILD_TESTS            OFF CACHE BOOL "" FORCE)
set(LIBIGL_BUILD_TUTORIALS        OFF CACHE BOOL "" FORCE)
set(LIBIGL_EMBREE                 OFF CACHE BOOL "" FORCE)
set(LIBIGL_GLFW                   OFF CACHE BOOL "" FORCE)
set(LIBIGL_IMGUI                  OFF CACHE BOOL "" FORCE)
set(LIBIGL_OPENGL                 OFF CACHE BOOL "" FORCE)
set(LIBIGL_PREDICATES             OFF CACHE BOOL "" FORCE)
set(LIBIGL_XML                    OFF CACHE BOOL "" FORCE)
set(LIBIGL_COPYLEFT_CGAL          OFF CACHE BOOL "" FORCE)
set(LIBIGL_COPYLEFT_TETGEN        OFF CACHE BOOL "" FORCE)
set(LIBIGL_RESTRICTED_MATLAB      OFF CACHE BOOL "" FORCE)
set(LIBIGL_RESTRICTED_MOSEK       OFF CACHE BOOL "" FORCE)
set(LIBIGL_RESTRICTED_TRIANGLE    OFF CACHE BOOL "" FORCE)

if(EXISTS "${CROSSLIFT_DEPS_DIR}/libigl-src/CMakeLists.txt")
  FetchContent_Declare(libigl SOURCE_DIR "${CROSSLIFT_DEPS_DIR}/libigl-src")
else()
  FetchContent_Declare(libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG        66b3ef2253e765d0ce0db74cec91bd706e5ba176)  # v2.4.0
endif()
FetchContent_MakeAvailable(libigl)

# ============================================================================
# OpenMesh 12.0-RC1 — required by libQEx
# ============================================================================
set(BUILD_APPS            OFF CACHE BOOL "" FORCE)
set(OPENMESH_DOCS         OFF CACHE BOOL "" FORCE)
set(OPENMESH_BUILD_SHARED OFF CACHE BOOL "" FORCE)

set(_openmesh_src   "${CROSSLIFT_DEPS_DIR}/openmesh-src")
set(_rwth_gitlab    "https://gitlab.vci.rwth-aachen.de:9000")
set(_openmesh_ref   d9b65de1de265d48120220c485134c4e7cb733c6)  # OpenMesh 12.0-RC1
set(_cmake_lib_ref  9515a4170bc55880e350d9d8ee1af67b77b4147d)  # cmake-library submodule

# Fallback: source archives via CMake's downloader when git cannot reach the remote.
if(NOT EXISTS "${_openmesh_src}/cmake-library/VCI/VCICommon.cmake")
  find_package(Git REQUIRED)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" ls-remote --exit-code "${_rwth_gitlab}/OpenMesh/OpenMesh.git" HEAD
    RESULT_VARIABLE _openmesh_git_rc TIMEOUT 30 OUTPUT_QUIET ERROR_QUIET)
  if(NOT _openmesh_git_rc EQUAL 0)
    message(STATUS "OpenMesh git remote unreachable; downloading source archives")
    FetchContent_Populate(openmesh_archive
      URL "${_rwth_gitlab}/OpenMesh/OpenMesh/-/archive/${_openmesh_ref}/OpenMesh-${_openmesh_ref}.tar.gz"
      SOURCE_DIR "${_openmesh_src}")
    FetchContent_Populate(openmesh_cmake_library
      URL "${_rwth_gitlab}/cmake/cmake-library/-/archive/${_cmake_lib_ref}/cmake-library-${_cmake_lib_ref}.tar.gz"
      SOURCE_DIR "${_openmesh_src}/cmake-library")
  endif()
endif()

if(EXISTS "${_openmesh_src}/CMakeLists.txt")
  FetchContent_Declare(openmesh SOURCE_DIR "${_openmesh_src}")
else()
  FetchContent_Declare(openmesh
    GIT_REPOSITORY "${_rwth_gitlab}/OpenMesh/OpenMesh.git"
    GIT_TAG        ${_openmesh_ref})
endif()
FetchContent_MakeAvailable(openmesh)

# ============================================================================
# libQEx — quad extraction. Built as an explicit target: its own CMakeLists
# uses a find_package(OpenMesh) form that predates the config package OpenMesh
# ships today, so driving the three source files directly is less fragile than
# patching it.
# ============================================================================
if(EXISTS "${CROSSLIFT_DEPS_DIR}/libqex-src/CMakeLists.txt")
  set(libqex_SOURCE_DIR "${CROSSLIFT_DEPS_DIR}/libqex-src")
else()
  FetchContent_Declare(libqex
    GIT_REPOSITORY https://github.com/hcebke/libQEx.git
    GIT_TAG        c0a4dc69d9b22685d07ea4c8f1ea4e92dd407264)
  FetchContent_GetProperties(libqex)
  if(NOT libqex_POPULATED)
    FetchContent_Populate(libqex)
  endif()
endif()

add_library(QEx STATIC
  ${libqex_SOURCE_DIR}/interfaces/c/qex.cc
  ${libqex_SOURCE_DIR}/src/predicates.c
  ${libqex_SOURCE_DIR}/src/MeshExtractor.cc
)
target_include_directories(QEx PUBLIC
  ${libqex_SOURCE_DIR}/interfaces/c
  ${libqex_SOURCE_DIR}/src
  ${openmesh_SOURCE_DIR}/src
)
target_link_libraries(QEx PUBLIC OpenMeshCoreStatic OpenMeshToolsStatic)
# predicates.c needs strict SSE semantics; x87 excess precision breaks the
# exact-arithmetic orientation tests it relies on.
target_compile_options(QEx PRIVATE -msse -mfpmath=sse)
target_compile_definitions(QEx PRIVATE QEX_EXPORT_SYMBOLS)

# ============================================================================
# Directional — seamless integration and streamline tracing (header-only)
# ============================================================================
if(EXISTS "${CROSSLIFT_DEPS_DIR}/directional-src/include/directional/streamlines.h")
  set(directional_SOURCE_DIR "${CROSSLIFT_DEPS_DIR}/directional-src")
else()
  FetchContent_Declare(directional
    GIT_REPOSITORY https://github.com/avaxman/Directional.git
    GIT_TAG        110fd7a8b1d9f0f3d79060a9ffb98d184c84378e)
  FetchContent_GetProperties(directional)
  if(NOT directional_POPULATED)
    FetchContent_Populate(directional)
  endif()
endif()

# ---------------------------------------------------------------------------
# Patch Directional in place.
#
# Two kinds of change, and it is worth keeping them straight:
#   (a) compatibility — Directional targets Eigen 3.4 / C++20, while libigl
#       v2.4.0 pins Eigen 3.3. Pure mechanical substitutions.
#   (b) genuine upstream bugs — see docs/directional-patches.md. These are
#       load-bearing: without them streamline tracing reads uninitialized
#       adjacency on any mesh with boundary, and seeded streamlines start off
#       the surface.
#
# Every replacement is written to be a no-op when already applied, so
# reconfiguring an existing checkout is safe.
# ---------------------------------------------------------------------------

# (a) sparse_diagonal.h uses Eigen::Vector, which is Eigen 3.4+.
file(READ "${directional_SOURCE_DIR}/include/directional/sparse_diagonal.h" _sd)
string(REPLACE "Eigen::Vector<Scalar, Eigen::Dynamic>"
               "Eigen::Matrix<Scalar, Eigen::Dynamic, 1>" _sd "${_sd}")
file(WRITE "${directional_SOURCE_DIR}/include/directional/sparse_diagonal.h" "${_sd}")

# (a) set_diff.h calls .begin()/.end() on Eigen vectors (Eigen 3.4 STL iterators).
file(READ "${directional_SOURCE_DIR}/include/directional/set_diff.h" _sdiff)
string(REPLACE "B.begin(), B.end()" "B.data(), B.data() + B.size()" _sdiff "${_sdiff}")
file(WRITE "${directional_SOURCE_DIR}/include/directional/set_diff.h" "${_sdiff}")

# (a) <numbers> / std::numbers::pi is C++20 and absent on GCC < 10.
set(_NUMBERS_HEADERS
  gaussian_curvature.h principal_matching.h effort_to_indices.h power_to_raw.h
  polyvector_to_raw.h IntrinsicVertexTangentBundle.h rotation_to_raw.h
  dual_cycles.h index_prescription.h)
foreach(_hdr ${_NUMBERS_HEADERS})
  set(_p "${directional_SOURCE_DIR}/include/directional/${_hdr}")
  if(EXISTS "${_p}")
    file(READ "${_p}" _c)
    string(REPLACE "#include <numbers>"
      "#include <cmath>\n#ifndef M_PI\n#define M_PI 3.14159265358979323846\n#endif" _c "${_c}")
    string(REPLACE "std::numbers::pi" "M_PI" _c "${_c}")
    file(WRITE "${_p}" "${_c}")
  endif()
endforeach()

# (b) TriMesh.h leaves triangle-triangle adjacency uninitialized, so boundary
#     edges hold garbage face indices that streamline tracing dereferences.
set(_tm "${directional_SOURCE_DIR}/include/directional/TriMesh.h")
if(EXISTS "${_tm}")
  file(READ "${_tm}" _c)
  string(REPLACE "TT.resize(F.rows(),3);" "TT = Eigen::MatrixXi::Constant(F.rows(),3,-1);"
         _c "${_c}")
  file(WRITE "${_tm}" "${_c}")
endif()

# (b) streamlines.cpp: two bugs.
#     1. Boundary edges are skipped outright (`if (f1==-1) continue;`). With the
#        TT fix above that no longer crashes, but it still drops the last
#        segment of any streamline that should end at the boundary. Removing
#        the skip lets the intersection test run; nextElements stays -1 and the
#        existing `(nextElements < 0)` branch finalizes and retires the line.
#     2. The seeded branch slices barycenters with a 1-element column index, so
#        start points come out (k,1) — the Y coordinate only — instead of (k,3).
set(_sl "${directional_SOURCE_DIR}/include/directional/streamlines.cpp")
if(EXISTS "${_sl}")
  file(READ "${_sl}" _c)
  string(REPLACE
    "                if (f1==-1)\n                 continue;  //boundary"
    "                // boundary edge: patched, do not skip — let the segment terminate"
    _c "${_c}")
  string(REPLACE
    "                        if (f1==-1)\n                         continue;  //boundary"
    "                        // boundary edge: patched, do not skip — let the segment terminate"
    _c "${_c}")
  string(REPLACE "Eigen::VectorXi one(1); one(0)=1;" "Eigen::VectorXi three(3); three<<0,1,2;"
         _c "${_c}")
  string(REPLACE "data.sampleFaces, one, data.samplePoints"
                 "data.sampleFaces, three, data.samplePoints" _c "${_c}")
  file(WRITE "${_sl}" "${_c}")
endif()

add_library(Directional INTERFACE)
target_include_directories(Directional INTERFACE ${directional_SOURCE_DIR}/include)

# ============================================================================
# The extension module
# ============================================================================
nanobind_add_module(_core
  src/cpp/bindings.cpp
  src/cpp/miq.cpp
  src/cpp/qex.cpp
  src/cpp/directional.cpp
)
target_include_directories(_core PRIVATE src/cpp/include)
target_link_libraries(_core PRIVATE igl::core igl_copyleft::comiso QEx Directional)

# Lands the .so beside the Python sources so `crosslift._core` resolves.
install(TARGETS _core LIBRARY DESTINATION crosslift)

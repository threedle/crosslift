from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from time import time

import numpy as np

from crosslift.geometry.io import read_quad_obj, write_obj

BIN_SUBDIR = os.path.join("build", "Build", "bin")
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _quadwild_root() -> str:
    candidates = []
    if os.environ.get("QUADWILD_DIR"):
        candidates.append(os.environ["QUADWILD_DIR"])
    candidates.append(str(_PROJECT_ROOT / "third_party" / "quadwild"))

    for root in candidates:
        if os.path.exists(os.path.join(root, BIN_SUBDIR, "quadwild")):
            return root
    raise RuntimeError(
        "QuadWild not found; looked in " + ", ".join(candidates) + ". Build it "
        "with scripts/install_quadwild.sh, or point QUADWILD_DIR at an existing "
        f"quadwild-bimdf checkout containing {BIN_SUBDIR}.")


def _quadwild_paths() -> dict[str, str]:
    root = _quadwild_root()
    bin_dir = os.path.join(root, BIN_SUBDIR)
    return {
        "root": root,
        "quadwild": os.path.join(bin_dir, "quadwild"),
        "quad_from_patches": os.path.join(bin_dir, "quad_from_patches"),
        "flow_config": os.path.join(root, "config", "main_config", "flow.txt"),
    }


def _write_rosy(path: str, PD1: np.ndarray) -> None:
    """Write a 4-RoSy field in vcglib's ASCII format, one PD1 per face."""
    with open(path, "w") as fh:
        fh.write(f"{PD1.shape[0]}\n4\n")
        fh.writelines(f"{v[0]} {v[1]} {v[2]}\n" for v in PD1)


def _write_sharp(path: str, entries) -> None:
    """Write a QuadWild ``.sharp`` file.

    Args:
        entries: ``(type, face, local_edge)`` triples, type 0 concave or
            1 convex, ``local_edge`` in 0..2.
    """
    with open(path, "w") as fh:
        fh.write(f"{len(entries)}\n")
        fh.writelines(f"{int(t)},{int(f)},{int(e)}\n" for t, f, e in entries)


def _write_basic_config(path: str, sharp_angle: float) -> None:
    """Write QuadWild's basic setup file at stock values, with remeshing off."""
    with open(path, "w") as fh:
        fh.write("do_remesh 0\n")
        fh.write(f"sharp_feature_thr {sharp_angle}\n")
        fh.write("alpha 0.01\n")
        fh.write("scaleFact 1\n")


def _flow_config(workdir: str, stock: str, root: str, scale_fact: float) -> str:
    """Write a copy of the ILP config into ``workdir``.

    ``scaleFact`` is overridden and the nested config filenames, which the stock
    file gives relative to the QuadWild root, are rewritten absolute.

    Args:
        workdir: directory the copy is written into.
        stock: the upstream ``flow.txt`` to copy.
        root: QuadWild checkout the relative filenames resolve against.
        scale_fact: value to write for ``scaleFact``.

    Returns:
        Path to the copy.
    """
    def absolute(match):
        key, value = match.group(1), match.group(2)
        if not os.path.isabs(value):
            value = os.path.normpath(os.path.join(root, value))
        return f'{key} "{value}"'

    with open(stock) as fh:
        text = fh.read()
    text = re.sub(r'^(flow_config_filename|satsuma_config_filename) "([^"]*)"',
                  absolute, text, flags=re.MULTILINE)
    text = re.sub(r"^scaleFact .*$", f"scaleFact {scale_fact}", text,
                  flags=re.MULTILINE)

    path = os.path.join(workdir, "flow.txt")
    with open(path, "w") as fh:
        fh.write(text)
    return path


def _run(cmd, cwd: str, name: str) -> None:
    """Run one QuadWild stage, relaying its output on failure."""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", check=False)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise RuntimeError(f"{name} exited {result.returncode}")


def sharp_edge_entries(V: np.ndarray, F: np.ndarray, angle: float):
    """QuadWild ``.sharp`` entries for the creased edges of a triangle mesh.

    Args:
        V: (nV, 3) vertex positions.
        F: (nF, 3) faces.
        angle: normal-deviation threshold in degrees. ``<= 0`` emits nothing.

    Returns:
        A list of ``(type, face, local_edge)`` triples, type fixed to 1.
    """
    if angle is None or angle <= 0:
        return []

    F = np.asarray(F, dtype=np.int64)
    corners = np.stack([F, np.roll(F, -1, axis=1)], axis=-1).reshape(-1, 2)
    _keys, inverse, counts = np.unique(
        np.sort(corners, axis=1), axis=0, return_inverse=True, return_counts=True)

    order = np.argsort(inverse.reshape(-1), kind="stable")
    starts = np.cumsum(counts) - counts
    interior = counts == 2
    first, second = order[starts[interior]], order[starts[interior] + 1]

    p = V[F]
    normals = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-16)

    face_of = np.repeat(np.arange(F.shape[0]), 3)
    local_of = np.tile(np.arange(3), F.shape[0])
    cos_theta = (normals[face_of[first]] * normals[face_of[second]]).sum(axis=1)
    sharp = first[np.arccos(np.clip(cos_theta, -1.0, 1.0)) > np.radians(angle)]

    return [(1, int(face_of[c]), int(local_of[c])) for c in sharp]


def run_quadwild(
    V: np.ndarray,
    F: np.ndarray,
    PD1: np.ndarray,
    output_path: str | None = None,
    *,
    sharp_angle: float = -1.0,
    scale_fact: float = 1.0,
    keep_workdir: bool = False,
    label: str = "  ",
) -> tuple[np.ndarray, np.ndarray]:
    """Quad-remesh a triangle mesh with QuadWild, conditioned on a cross field.

    Args:
        V: (nV, 3) float64 vertex positions.
        F: (nF, 3) int32 faces.
        PD1: (nF, 3) float64 first cross-field direction per face.
        output_path: where to copy the final quad OBJ, or None to skip.
        sharp_angle: normal-deviation threshold in degrees for sharp features,
            used both for the ``.sharp`` file and QuadWild's own detection.
            ``<= 0`` disables sharp features.
        scale_fact: flow config ``scaleFact``; smaller gives denser quads. 1.0
            passes the stock config through.
        keep_workdir: leave the scratch directory on disk.
        label: prefix for the progress lines.

    Returns:
        ``(quad_V, quad_F)``.
    """
    paths = _quadwild_paths()
    for key in ("quadwild", "quad_from_patches", "flow_config"):
        if not os.path.exists(paths[key]):
            raise RuntimeError(
                f"QuadWild {key} not found at {paths[key]}; the install at "
                f"{paths['root']} is incomplete. Rerun scripts/install_quadwild.sh.")
    if PD1.shape[0] != F.shape[0]:
        raise ValueError(
            f"field has {PD1.shape[0]} faces but the mesh has {F.shape[0]}")

    workdir = tempfile.mkdtemp(prefix="quadwild_")
    try:
        mesh_path = os.path.join(workdir, "mesh.obj")
        rosy_path = os.path.join(workdir, "mesh.rosy")
        config_path = os.path.join(workdir, "basic_setup.txt")
        sharp_path = os.path.join(workdir, "mesh.sharp")

        write_obj(mesh_path, V, F)
        _write_rosy(rosy_path, PD1)
        _write_basic_config(config_path, sharp_angle)

        entries = sharp_edge_entries(V, F, sharp_angle)
        cmd = [paths["quadwild"], mesh_path, "2", config_path, rosy_path]
        if entries:
            _write_sharp(sharp_path, entries)
            cmd.append(sharp_path)
            print(f"{label}{len(entries)} sharp edge(s) at {sharp_angle:g} deg")
        else:
            print(f"{label}sharp features disabled")

        print(f"{label}QuadWild field + tracing")
        start = time()
        _run(cmd, workdir, "quadwild")
        print(f"{label}  {time() - start:.2f}s")

        traced = os.path.join(workdir, "mesh_rem_p0.obj")
        if not os.path.exists(traced):
            raise RuntimeError(f"quadwild produced no {os.path.basename(traced)}")

        flow = _flow_config(workdir, paths["flow_config"], paths["root"], scale_fact)
        print(f"{label}QuadWild ILP quantization (scaleFact {scale_fact:g})")
        start = time()
        _run([paths["quad_from_patches"], traced, "1", flow],
             workdir, "quad_from_patches")
        print(f"{label}  {time() - start:.2f}s")

        quads = os.path.join(workdir, "mesh_rem_p0_1_quadrangulation_smooth.obj")
        if not os.path.exists(quads):
            raise RuntimeError(
                f"quad_from_patches produced no {os.path.basename(quads)}")

        quad_V, quad_F = read_quad_obj(quads)
        print(f"{label}  quad vertices: {quad_V.shape[0]}, "
              f"quad faces: {quad_F.shape[0]}")
        if output_path:
            shutil.copyfile(quads, output_path)
            print(f"{label}Saved quad mesh: {output_path}")
        return quad_V, quad_F
    finally:
        if keep_workdir:
            print(f"{label}kept workdir: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

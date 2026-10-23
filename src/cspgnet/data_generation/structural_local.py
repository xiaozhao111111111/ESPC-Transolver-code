"""Calibration-only graded P2 DtN for structural block solids.

The frozen v3 geometry/numerical kernels are not edited.  This extension
uses their consistent reaction/Gram projection, with a new CAD factory.
Passing this module's local checks does not authorize scene production.
"""
from __future__ import annotations
import time
import numpy as np
from ngsolve import Conj
from . import formal_neumann_3d as formal
from . import tmatrix_neumann_3d as tmatrix
from .structural_voxels import StructuralSpec, make_solid, audit_surface


def build_mesh(spec: StructuralSpec, profile: tmatrix.GradedMeshProfile):
    if profile.name not in tmatrix.GRADED_MESH_PROFILES or profile != tmatrix.GRADED_MESH_PROFILES[profile.name]:
        raise ValueError("only registered graded profiles can produce calibration meshes")
    obstacle = make_solid(spec)
    outer = formal.Sphere((0, 0, 0), 1.5)
    for face in outer.faces:
        face.name = "canonical"
        face.maxh = profile.canonical_h
    for face in obstacle.faces:
        face.name = "scatterer"
        face.maxh = profile.scatterer_h
    mesh = formal.Mesh(formal.OCCGeometry(outer - obstacle).GenerateMesh(
        maxh=profile.background_h, grading=profile.grading))
    mesh.Curve(2)
    return mesh


def audit_cad_mesh(spec: StructuralSpec, mesh) -> dict:
    points = np.asarray([tuple(point.p) for point in mesh.ngmesh.Points()], dtype=np.float64)
    boundaries = tuple(mesh.GetBoundaries())
    faces = np.asarray([[vertex.nr - 1 for vertex in element.vertices]
                        for element in mesh.ngmesh.Elements2D()
                        if boundaries[element.index - 1] == "scatterer"], dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError("missing triangular scatterer boundary")
    a, b, c = (points[faces[:, index]] for index in range(3))
    cross = np.cross(b - a, c - a)
    length = np.linalg.norm(cross, axis=1)
    if np.any(length <= 1.0e-14):
        raise RuntimeError("degenerate CAD boundary triangle")
    normal = cross / length[:, None]
    midpoint = (a + b + c) / 3
    delta = spec.cell_width * 1.0e-4
    plus = spec.contains(midpoint + delta * normal, tolerance=1.0e-12)
    minus = spec.contains(midpoint - delta * normal, tolerance=1.0e-12)
    if np.any(plus == minus):
        raise RuntimeError("CAD boundary normal fails inside/outside perturbation")
    reverse = plus & ~minus
    faces[reverse, 1], faces[reverse, 2] = faces[reverse, 2].copy(), faces[reverse, 1].copy()
    audit = audit_surface(points, faces)
    used = points[np.unique(faces)]
    radius = float(np.linalg.norm(used, axis=1).max())
    expected_volume = len(spec.cells) * spec.cell_width**3
    extra = {"expected_genus": audit["genus"] == spec.genus,
             "enclosed_by_radius_0_9": radius < .9,
             "matches_manifest_volume": np.isclose(audit["oriented_volume"], expected_volume, rtol=1.0e-7)}
    audit["checks"].update({key: bool(value) for key, value in extra.items()})
    audit["passed"] = all(audit["checks"].values())
    audit.update(bounding_radius=radius, expected_volume=expected_volume,
                 verified_outward_normals=True)
    return audit


def _problem(spec, profile):
    mesh = build_mesh(spec, profile)
    metadata = {**tmatrix._mesh_diagnostics(mesh), "cad_audit": audit_cad_mesh(spec, mesh)}
    if not metadata["cad_audit"]["passed"]:
        raise RuntimeError(f"actual CAD/topology differs from design: {metadata}")
    fes = formal.H1(mesh, order=2, complex=True, dirichlet="canonical")
    trial, test = fes.TnT()
    form = formal.BilinearForm(fes)
    form += formal.grad(trial) * formal.grad(test) * formal.dx
    form += -4.0 * trial * test * formal.dx
    form.Assemble()
    inverse = form.mat.Inverse(freedofs=fes.FreeDofs())
    mass = formal.BilinearForm(fes)
    mass += trial * test * formal.ds("canonical")
    mass.Assemble()
    boundary = np.flatnonzero(np.asarray(fes.GetDofs(mesh.Boundaries("canonical"))))
    free = np.asarray(fes.FreeDofs(), dtype=bool)
    traces = np.empty((fes.ndof, 121), dtype=np.complex128)
    conjugate = np.empty((len(boundary), 121), dtype=np.complex128)
    mass_traces = np.empty_like(conjugate)
    trace = formal.GridFunction(fes)
    input_vec, mass_vec = trace.vec.CreateVector(), trace.vec.CreateVector()
    for column, (ell, order) in enumerate(formal.spherical_modes(10)):
        harmonic = formal.SphericalHarmonicsCF(ell)
        harmonic.sh[ell, order] = 1.0
        trace.Set(harmonic, definedon=mesh.Boundaries("canonical"))
        traces[:, column] = trace.vec.FV().NumPy()
        input_vec.FV().NumPy()[:] = traces[:, column]
        mass_vec.data = mass.mat * input_vec
        mass_traces[:, column] = mass_vec.FV().NumPy()[boundary]
        trace.Set(Conj(harmonic), definedon=mesh.Boundaries("canonical"))
        conjugate[:, column] = trace.vec.FV().NumPy()[boundary]
    gram = conjugate.T @ mass_traces
    metadata.update(fes_dofs=fes.ndof, boundary_dofs=len(boundary),
                    gram_condition=float(np.linalg.cond(gram)),
                    mesh_profile=profile.as_dict(), mesh_strategy="graded-local-FEM",
                    k=2.0, fem_order=2, curved_mesh=True)
    return mesh, fes, form, inverse, boundary, free, traces, conjugate, gram, metadata


def modal_dtn(spec: StructuralSpec, profile: tmatrix.GradedMeshProfile):
    started = time.perf_counter()
    mesh, fes, form, inverse, boundary, free, traces, conjugate, gram, meta = _problem(spec, profile)
    field = formal.GridFunction(fes)
    correction, reaction = field.vec.CreateVector(), field.vec.CreateVector()
    measurements = np.empty((121, 100), dtype=np.complex128)
    residuals = []
    for column in range(100):
        field.vec.FV().NumPy()[:] = traces[:, column]
        correction.data = -form.mat * field.vec
        field.vec.data += inverse * correction
        reaction.data = form.mat * field.vec
        values = reaction.FV().NumPy()
        residuals.append(float(np.linalg.norm(values[free]) / max(np.linalg.norm(values), 1.0e-15)))
        measurements[:, column] = conjugate.T @ values[boundary]
    matrix = np.linalg.solve(gram, measurements)
    if not np.isfinite(matrix).all():
        raise FloatingPointError("non-finite structural DtN")
    meta.update(max_free_dof_relative_residual=max(residuals), seconds=time.perf_counter()-started,
                input_lmax=9, output_lmax=10)
    return matrix, meta


def direct_flux(spec: StructuralSpec, profile: tmatrix.GradedMeshProfile, coefficients):
    """Fresh FEM assembly, actual boundary CF, no modal operator multiplication."""
    started = time.perf_counter()
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    if coefficients.ndim == 1:
        coefficients = coefficients[:, None]
    if coefficients.shape[0] != 100:
        raise ValueError("direct reference requires 100 coefficients (zero-pad truncated traces)")
    mesh, fes, form, inverse, boundary, free, traces, conjugate, gram, meta = _problem(spec, profile)
    field = formal.GridFunction(fes)
    correction, reaction = field.vec.CreateVector(), field.vec.CreateVector()
    measurements = np.empty((121, coefficients.shape[1]), dtype=np.complex128)
    residuals = []
    for column in range(coefficients.shape[1]):
        harmonic = formal.SphericalHarmonicsCF(9)
        for value, (ell, order) in zip(coefficients[:, column], formal.spherical_modes(9)):
            harmonic.sh[ell, order] = complex(value)
        field.Set(harmonic, definedon=mesh.Boundaries("canonical"))
        correction.data = -form.mat * field.vec
        field.vec.data += inverse * correction
        reaction.data = form.mat * field.vec
        values = reaction.FV().NumPy()
        residuals.append(float(np.linalg.norm(values[free]) / max(np.linalg.norm(values), 1.0e-15)))
        measurements[:, column] = conjugate.T @ values[boundary]
    result = np.linalg.solve(gram, measurements)
    meta.update(max_free_dof_relative_residual=max(residuals), seconds=time.perf_counter()-started)
    return result, meta

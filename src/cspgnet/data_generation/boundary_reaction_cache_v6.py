"""Unprojected FEM reaction reuse for independent input-truncation auditing."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


def file_sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024**2),b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class BoundaryReactionCache:
    reactions: np.ndarray
    mass: csc_matrix
    boundary_dofs: np.ndarray
    boundary_traces: np.ndarray
    conjugate_boundary_traces: np.ndarray
    gram: np.ndarray
    receipt: dict

    def __post_init__(self):
        n,count=self.reactions.shape
        if count!=100 or self.mass.shape!=(n,n) or self.boundary_dofs.shape!=(n,):
            raise ValueError("reaction cache dimensions are inconsistent")
        if self.boundary_traces.shape!=(n,121) or self.conjugate_boundary_traces.shape!=(n,121):
            raise ValueError("reaction cache must use the full 121 output test traces")
        if self.gram.shape!=(121,121):
            raise ValueError("Gram matrix must be [121,121]")
        if len(np.unique(self.boundary_dofs))!=n or np.any(self.boundary_dofs<0):
            raise ValueError("boundary DOF mapping is not injective")
        for values in (self.reactions,self.mass.data,self.boundary_traces,
                       self.conjugate_boundary_traces,self.gram):
            if not np.isfinite(values).all():
                raise ValueError("nonfinite reaction cache")
        self.mass=self.mass.astype(np.complex128).tocsc()
        self._mass_factor=splu(self.mass)

    @classmethod
    def load(cls, directory: Path):
        directory=Path(directory).resolve()
        receipt=json.loads((directory/"boundary_artifact_receipt.json").read_text(encoding="utf-8"))
        if receipt.get("schema")!="structural-smooth-v6-full-boundary-reactions-v1":
            raise ValueError("unsupported full boundary reaction cache schema")
        paths={}
        for kind in ("artifact","mesh"):
            path=(directory/receipt[kind]).resolve()
            if path.parent!=directory or not path.is_file() or file_sha256(path)!=receipt[kind+"_sha256"]:
                raise ValueError(f"boundary cache {kind} path/hash mismatch")
            paths[kind]=path
        if (receipt["input_lmax"]!=9 or receipt["output_lmax"]!=10 or receipt["fem_order"]!=2
                or receipt["k"]!=2.0 or receipt["canonical_radius"]!=1.5):
            raise ValueError("boundary cache physics/modal configuration mismatch")
        with np.load(paths["artifact"],allow_pickle=False) as archive:
            mass=csc_matrix((archive["mass_data"],archive["mass_indices"],archive["mass_indptr"]),
                            shape=tuple(int(x) for x in archive["mass_shape"]))
            instance=cls(np.asarray(archive["reaction_modes"],dtype=np.complex128),mass,
                         np.asarray(archive["boundary_dofs"],dtype=np.int64),
                         np.asarray(archive["boundary_traces"],dtype=np.complex128),
                         np.asarray(archive["conjugate_boundary_traces"],dtype=np.complex128)
                         if "conjugate_boundary_traces" in archive else
                         np.conjugate(archive["boundary_traces"])
                         if receipt.get("conjugate_trace_encoding")=="exact-conjugate-verified" else
                         _missing_conjugate_traces(),
                         np.asarray(archive["gram"],dtype=np.complex128),receipt)
        if instance.reactions.shape!=tuple(receipt["reaction_shape"]):
            raise ValueError("receipt/array reaction shape mismatch")
        return instance

    def reaction_from_coefficients(self, coefficients):
        coefficients=np.asarray(coefficients,dtype=np.complex128)
        if coefficients.ndim not in (1,2) or coefficients.shape[0] not in (16,36,64,100):
            raise ValueError("coefficients must have 16,36,64,100 rows")
        if not np.isfinite(coefficients).all():
            raise ValueError("nonfinite projection coefficients")
        return self.reactions[:,:coefficients.shape[0]] @ coefficients

    def flux_from_reaction(self, reaction):
        reaction=np.asarray(reaction,dtype=np.complex128)
        if reaction.shape[0]!=self.mass.shape[0] or not np.isfinite(reaction).all():
            raise ValueError("full reaction must match the exact cached boundary DOFs")
        return self._mass_factor.solve(reaction)

    def weighted_energy(self, flux):
        flux=np.asarray(flux,dtype=np.complex128)
        return float(np.vdot(flux,self.mass @ flux).real)

    def compare_independent_full_reaction(self, full_reaction, coefficients):
        """Caller supplies a separately solved, untruncated GT-trace reaction."""
        full=self.flux_from_reaction(full_reaction)
        truncated=self.flux_from_reaction(self.reaction_from_coefficients(coefficients))
        denominator=self.weighted_energy(full)
        if denominator<=0:
            raise ValueError("zero/invalid reference flux energy requires an absolute-error report")
        delta_energy=max(self.weighted_energy(full-truncated),0.0)
        full_modal=np.linalg.solve(self.gram,self.conjugate_boundary_traces.T @ full_reaction)
        truncated_reaction=self.reaction_from_coefficients(coefficients)
        truncated_modal=np.linalg.solve(self.gram,self.conjugate_boundary_traces.T @ truncated_reaction)
        projected_full=self.boundary_traces @ full_modal
        projected_delta=self.boundary_traces @ (full_modal-truncated_modal)
        projected_denominator=self.weighted_energy(projected_full)
        if projected_denominator<=0:
            raise ValueError("zero projected reference energy")
        return {
            "input_truncation_full_flux_weighted_relative":float(np.sqrt(delta_energy/denominator)),
            "input_truncation_output_l10_weighted_relative":float(np.sqrt(
                max(self.weighted_energy(projected_delta),0.0)/projected_denominator)),
            "independent_full_flux_output_l10_projection_relative":float(np.sqrt(
                max(self.weighted_energy(full-projected_full),0.0)/denominator)),
            "full_reference_energy":denominator,
            "reference_source":"caller-supplied independent direct FEM GT-trace reaction",
        }


def _missing_conjugate_traces():
    raise ValueError("conjugate traces missing without a verified lossless encoding")


def compact_copy_boundary_artifacts(source: Path, target: Path) -> dict:
    """Remove bitwise duplicate arrays, retaining exact reactions/basis/mass/mesh."""
    source=Path(source).resolve()
    target=Path(target).resolve()
    cache=BoundaryReactionCache.load(source)
    if target.exists():
        raise FileExistsError("compact reaction cache target must be new")
    target.mkdir(parents=True)
    conjugate_exact=np.array_equal(cache.conjugate_boundary_traces,np.conjugate(cache.boundary_traces))
    fields={"reaction_modes":cache.reactions,"boundary_dofs":cache.boundary_dofs,
            "boundary_traces":cache.boundary_traces,"gram":cache.gram,
            "mass_data":cache.mass.data,"mass_indices":cache.mass.indices,
            "mass_indptr":cache.mass.indptr,"mass_shape":np.asarray(cache.mass.shape,dtype=np.int64)}
    if not conjugate_exact:
        fields["conjugate_boundary_traces"]=cache.conjugate_boundary_traces
    artifact=target/"boundary_reactions.npz"
    pending=target/"boundary_reactions.pending.npz"
    with pending.open("wb") as stream:
        np.savez_compressed(stream,**fields)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending,artifact)
    mesh=target/"production_mesh.vol.gz"
    shutil.copyfile(source/cache.receipt["mesh"],mesh)
    receipt={**cache.receipt,"artifact":artifact.name,"artifact_sha256":file_sha256(artifact),
             "mesh":mesh.name,"mesh_sha256":file_sha256(mesh),"artifact_bytes":artifact.stat().st_size,
             "conjugate_trace_encoding":"exact-conjugate-verified" if conjugate_exact else "stored-explicitly",
             "source_cache_receipt_sha256":file_sha256(source/"boundary_artifact_receipt.json"),
             "source_artifact_sha256":cache.receipt["artifact_sha256"],
             "compression_is_lossless":True}
    receipt_path=target/"boundary_artifact_receipt.json"
    receipt_path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    restored=BoundaryReactionCache.load(target)
    if not (np.array_equal(restored.reactions,cache.reactions)
            and np.array_equal(restored.boundary_traces,cache.boundary_traces)
            and np.array_equal(restored.conjugate_boundary_traces,cache.conjugate_boundary_traces)):
        raise ValueError("lossless boundary-cache roundtrip failed")
    return receipt

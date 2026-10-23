"""Experimental reusable-factor backends for batches of FEM Dirichlet lifts.

Experimental backends include explicit full-matrix COO/CSC extraction plus
SciPy factorization, NGSolve reduced SuperLU with dense matrix RHS, and a shared
BDDC preconditioner with per-RHS GMRES. Only BDDC-GMRES has passed one fixed
genus-0 end-to-end screen; no backend here is yet production-qualified across
the geometry family. The native scalar NGSolve factorization remains default.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.sparse import csc_matrix, issparse
from scipy.sparse.linalg import SuperLU, splu


@dataclass(frozen=True)
class BlockSolveResult:
    total_fields: np.ndarray
    reactions: np.ndarray
    relative_free_residuals: np.ndarray
    solve_seconds: float
    rhs_count: int
    batch_size: int


class BatchedReducedSolver:
    """Solve ``A_ff x = -(A u_lift)_f`` with one reusable sparse LU.

    ``A`` is the full assembled operator and ``free_dofs`` is the NGSolve
    Boolean mask.  Prescribed values are already present in ``lifting``;
    corrections are applied only to free DOFs.  The full sparse matrix is
    retained so callers can reuse the same LU for modal traces and independent
    flux traces on the same mesh.
    """

    def __init__(self, matrix, free_dofs: np.ndarray):
        if not issparse(matrix):
            raise TypeError("matrix must be a SciPy sparse matrix")
        self.matrix = csc_matrix(matrix, dtype=np.complex128)
        mask = np.asarray(free_dofs, dtype=bool)
        if mask.ndim != 1 or mask.size != self.matrix.shape[0]:
            raise ValueError("free_dofs must be a Boolean mask matching matrix rows")
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("operator matrix must be square")
        self.free_dofs = mask
        self.free_indices = np.flatnonzero(mask)
        if self.free_indices.size == 0:
            raise ValueError("operator has no free degrees of freedom")

        factor_started = time.perf_counter()
        reduced = self.matrix[self.free_indices, :][:, self.free_indices].tocsc()
        self.factor: SuperLU = splu(reduced)
        self.factorization_seconds = time.perf_counter() - factor_started
        self.reduced_nnz = int(reduced.nnz)
        self.matrix_nnz = int(self.matrix.nnz)
        del reduced

    @classmethod
    def from_ngsolve_matrix(cls, matrix, free_dofs: np.ndarray) -> "BatchedReducedSolver":
        """Extract a CSC matrix from an assembled NGSolve ``BaseMatrix``."""
        rows, cols, values = matrix.COO()
        shape = (int(matrix.height), int(matrix.width))
        sparse = csc_matrix((np.asarray(values, dtype=np.complex128),
                             (np.asarray(rows), np.asarray(cols))), shape=shape)
        return cls(sparse, free_dofs)

    def solve_lifts(self, lifting: np.ndarray, *, batch_size: int | None = None
                    ) -> BlockSolveResult:
        """Correct one or more lifts and return FEM reactions for every RHS.

        ``lifting`` has shape ``[ndof, nrhs]``. A batch size of ``None`` uses
        one matrix solve for all columns; finite batch sizes bound temporary
        dense memory without refactorizing the operator.
        """
        lifts = np.asarray(lifting, dtype=np.complex128)
        if lifts.ndim == 1:
            lifts = lifts[:, None]
        if lifts.ndim != 2 or lifts.shape[0] != self.matrix.shape[0]:
            raise ValueError("lifting must have shape [ndof, nrhs]")
        rhs_count = int(lifts.shape[1])
        if rhs_count == 0:
            raise ValueError("at least one right-hand side is required")
        if batch_size is None:
            chunk_size = rhs_count
        else:
            chunk_size = int(batch_size)
            if chunk_size <= 0:
                raise ValueError("batch_size must be positive")
            chunk_size = min(chunk_size, rhs_count)

        started = time.perf_counter()
        total = np.array(lifts, copy=True, order="F")
        for first in range(0, rhs_count, chunk_size):
            last = min(first + chunk_size, rhs_count)
            lift_block = lifts[:, first:last]
            rhs = -(self.matrix[self.free_indices, :] @ lift_block)
            correction = self.factor.solve(np.asfortranarray(rhs))
            total[self.free_indices, first:last] += correction

        reactions = np.asarray(self.matrix @ total, dtype=np.complex128)
        free_norm = np.linalg.norm(reactions[self.free_indices, :], axis=0)
        full_norm = np.linalg.norm(reactions, axis=0)
        residuals = free_norm / np.maximum(full_norm, 1.0e-30)
        return BlockSolveResult(
            total_fields=total,
            reactions=reactions,
            relative_free_residuals=residuals,
            solve_seconds=time.perf_counter() - started,
            rhs_count=rhs_count,
            batch_size=chunk_size,
        )


class NGSolveSuperLUBlockSolver:
    """Use NGSolve's reduced matrix and one exposed SuperLU factor for RHS blocks.

    Unlike :class:`BatchedReducedSolver`, this does not extract/re-factor the
    full FEM matrix. NGSolve constructs the reduced free-DOF factorization once;
    the underlying SciPy SuperLU ``solve`` is then called with a dense matrix
    of right-hand sides. NGSolve sparse matrix--MultiVector products form the
    residual and reaction blocks without a Python loop over matrix-vector
    products.
    """

    def __init__(self, operator_matrix, free_dofs):
        # Import registers the Python SuperLU inverse type with NGSolve.
        from ngsolve.directsolvers import SuperLU as _RegisteredSuperLU  # noqa: F401

        self.operator_matrix = operator_matrix
        self.ngsolve_free_dofs = free_dofs
        self.free_mask = np.asarray(free_dofs, dtype=bool)
        if self.free_mask.ndim != 1 or self.free_mask.size != int(operator_matrix.height):
            raise ValueError("free_dofs must match the assembled operator dimension")
        self.free_indices = np.flatnonzero(self.free_mask)
        if not self.free_indices.size:
            raise ValueError("operator has no free degrees of freedom")
        factor_started = time.perf_counter()
        self.factor = operator_matrix.Inverse(
            freedofs=self.ngsolve_free_dofs, inverse="superlu"
        )
        self.factorization_seconds = time.perf_counter() - factor_started
        self.superlu_factor = getattr(self.factor, "inv_mat", None)
        if self.superlu_factor is None or not callable(getattr(self.superlu_factor, "solve", None)):
            raise RuntimeError("NGSolve SuperLU factor does not expose matrix-RHS solve")

    @staticmethod
    def _multivector(operator_matrix, values: np.ndarray):
        import ngsolve as ng

        values = np.asarray(values, dtype=np.complex128)
        vectors = ng.MultiVector(operator_matrix.CreateVector(), values.shape[1])
        for column in range(values.shape[1]):
            vectors[column].FV().NumPy()[:] = values[:, column]
        return vectors

    @staticmethod
    def _dense_from_multivector(vectors, columns: int) -> np.ndarray:
        return np.column_stack([
            vectors[column].FV().NumPy() for column in range(columns)
        ])

    def solve_lifts(self, lifting: np.ndarray, *, batch_size: int | None = None
                    ) -> BlockSolveResult:
        """Correct Dirichlet lifts with dense multi-RHS triangular solves."""
        lifts = np.asarray(lifting, dtype=np.complex128)
        if lifts.ndim == 1:
            lifts = lifts[:, None]
        if lifts.ndim != 2 or lifts.shape[0] != self.free_mask.size:
            raise ValueError("lifting must have shape [ndof, nrhs]")
        rhs_count = int(lifts.shape[1])
        if rhs_count == 0:
            raise ValueError("at least one right-hand side is required")
        if batch_size is None:
            chunk_size = rhs_count
        else:
            chunk_size = int(batch_size)
            if chunk_size <= 0:
                raise ValueError("batch_size must be positive")
            chunk_size = min(chunk_size, rhs_count)

        started = time.perf_counter()
        total = np.array(lifts, copy=True, order="F")
        for first in range(0, rhs_count, chunk_size):
            last = min(first + chunk_size, rhs_count)
            block = lifts[:, first:last]
            lift_vectors = self._multivector(self.operator_matrix, block)
            rhs_vectors = (self.operator_matrix * lift_vectors).Evaluate()
            rhs_full = self._dense_from_multivector(rhs_vectors, last - first)
            rhs_free = -rhs_full[self.free_indices, :]
            correction = np.asarray(
                self.superlu_factor.solve(np.asfortranarray(rhs_free)),
                dtype=np.complex128,
            )
            if correction.ndim == 1:
                correction = correction[:, None]
            total[self.free_indices, first:last] += correction

        solution_vectors = self._multivector(self.operator_matrix, total)
        reaction_vectors = (self.operator_matrix * solution_vectors).Evaluate()
        reactions = self._dense_from_multivector(reaction_vectors, rhs_count)
        residuals = np.linalg.norm(reactions[self.free_indices, :], axis=0) / np.maximum(
            np.linalg.norm(reactions, axis=0), 1.0e-30
        )
        return BlockSolveResult(
            total_fields=total,
            reactions=reactions,
            relative_free_residuals=residuals,
            solve_seconds=time.perf_counter() - started,
            rhs_count=rhs_count,
            batch_size=chunk_size,
        )


class NGSolveBDDCGMRESBlockSolver:
    """Memory-light shared-preconditioner solves for repeated FEM lifts.

    The BDDC preconditioner must be constructed on the BilinearForm before its
    assembly and passed here after ``Update()``. Each RHS is solved with GMRES;
    the full FEM free-DOF residual is checked after the correction. This is an
    experimental low-memory alternative, not a production default.
    """

    def __init__(self, operator_matrix, preconditioner_matrix, free_dofs,
                 *, tolerance: float = 1.0e-10, max_steps: int = 400):
        self.operator_matrix = operator_matrix
        self.preconditioner_matrix = preconditioner_matrix
        self.free_mask = np.asarray(free_dofs, dtype=bool)
        if self.free_mask.ndim != 1 or self.free_mask.size != int(operator_matrix.height):
            raise ValueError("free_dofs must match the assembled operator dimension")
        if not np.any(self.free_mask):
            raise ValueError("operator has no free degrees of freedom")
        self.tolerance = float(tolerance)
        self.max_steps = int(max_steps)
        if self.tolerance <= 0 or self.max_steps <= 0:
            raise ValueError("GMRES tolerance and max_steps must be positive")

    def solve_lifts(self, lifting: np.ndarray, *, batch_size: int | None = None
                    ) -> BlockSolveResult:
        import ngsolve as ng

        lifts = np.asarray(lifting, dtype=np.complex128)
        if lifts.ndim == 1:
            lifts = lifts[:, None]
        if lifts.ndim != 2 or lifts.shape[0] != self.free_mask.size:
            raise ValueError("lifting must have shape [ndof, nrhs]")
        rhs_count = int(lifts.shape[1])
        if rhs_count == 0:
            raise ValueError("at least one right-hand side is required")
        if batch_size is None:
            chunk_size = rhs_count
        else:
            chunk_size = int(batch_size)
            if chunk_size <= 0:
                raise ValueError("batch_size must be positive")
            chunk_size = min(chunk_size, rhs_count)

        started = time.perf_counter()
        total = np.array(lifts, copy=True, order="F")
        reactions = np.empty_like(total)
        residuals = np.empty(rhs_count, dtype=np.float64)
        matrix = self.operator_matrix
        for first in range(0, rhs_count, chunk_size):
            last = min(first + chunk_size, rhs_count)
            for column in range(first, last):
                lift_vector = matrix.CreateColVector()
                lift_vector.FV().NumPy()[:] = lifts[:, column]
                full_rhs = -(matrix * lift_vector).Evaluate().FV().NumPy().copy()
                full_rhs[~self.free_mask] = 0.0
                rhs_vector = matrix.CreateColVector()
                rhs_vector.FV().NumPy()[:] = full_rhs
                correction = ng.solvers.GMRes(
                    matrix, rhs_vector, pre=self.preconditioner_matrix,
                    maxsteps=self.max_steps, tol=self.tolerance,
                    printrates=False,
                )
                total[:, column] += correction.FV().NumPy()
                total_vector = matrix.CreateColVector()
                total_vector.FV().NumPy()[:] = total[:, column]
                reaction = (matrix * total_vector).Evaluate().FV().NumPy().copy()
                reactions[:, column] = reaction
                residuals[column] = np.linalg.norm(reaction[self.free_mask]) / max(
                    np.linalg.norm(reaction), 1.0e-30
                )
                if residuals[column] > 1.0e-8:
                    raise RuntimeError(
                        f"BDDC-GMRES free-DOF residual {residuals[column]:.3e} "
                        f"exceeds 1e-8 for RHS {column}"
                    )

        return BlockSolveResult(
            total_fields=total,
            reactions=reactions,
            relative_free_residuals=residuals,
            solve_seconds=time.perf_counter() - started,
            rhs_count=rhs_count,
            batch_size=chunk_size,
        )

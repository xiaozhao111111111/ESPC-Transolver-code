"""NumPy-only S(Z), exactly the saved-Z information budget, no 100-mode cache."""
from __future__ import annotations
import numpy as np
from . import formal_neumann_3d as formal
from . import tmatrix_neumann_3d as tmatrix
from .reduced_z64_solver import modal_dtn_from_stored_z


def scattering_response(data):
    b,projection=modal_dtn_from_stored_z(data)
    points=np.asarray(data['interface_points'],float); radius=projection['radius']; k=float(data['k'])
    phi=formal.spherical_harmonic_matrix(points,formal.spherical_modes(7))
    h,_=tmatrix.radial_values(k,radius,7,outgoing=True)
    waves=phi*h[None]
    result=[]
    for modal in b:
        t,meta=tmatrix.transition_matrix(modal,k=k,canonical_radius=radius,lmax=7)
        if meta['conversion_residual']>1e-10: raise RuntimeError('saved-Z response conversion failed')
        result.append(waves@t)
    output=np.asarray(result,dtype=np.complex64)
    if not np.isfinite(output).all(): raise FloatingPointError('S(Z) not finite')
    return output

"""Independent structural FEM trace/flux and rotated-remeshing checks.

The archived structural_local implementation is preserved. This separate
extension shares its assembled weak formulation but not its stored DtN.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from . import structural_local as local
from . import tmatrix_neumann_3d as tmatrix
from . import formal_neumann_3d as formal
from .structural_voxels import make_solid


def direct_traces(spec, profile, coefficients, *, input_lmax):
    """Fresh FEM, actual SH coefficient functions, and full consistent-mass flux.

Returns projected l<=10 flux AND its L2 discrepancy to the unprojected
boundary flux M_bb^-1 r_b. This is distinct from derived training q64.
"""
    started=time.perf_counter()
    coefficients=np.asarray(coefficients,dtype=np.complex128)
    if coefficients.ndim==1: coefficients=coefficients[:,None]
    if coefficients.shape[0]!=(input_lmax+1)**2:
        raise ValueError('input coefficients must contain complete harmonic orders')
    mesh,fes,form,inverse,boundary,free,traces,conjugate,gram,meta=local._problem(spec,profile)
    trial,test=fes.TnT()
    mass=formal.BilinearForm(fes); mass+=trial*test*formal.ds('canonical'); mass.Assemble()
    row,col,value=mass.mat.COO()
    whole=coo_matrix((np.asarray(value),(np.asarray(row),np.asarray(col))),shape=(fes.ndof,fes.ndof)).tocsc()
    mb=whole[boundary,:][:,boundary].tocsc()
    mass_inverse=splu(mb)
    field=formal.GridFunction(fes)
    correction,reaction=field.vec.CreateVector(),field.vec.CreateVector()
    result=np.empty((121,coefficients.shape[1]),dtype=np.complex128)
    residuals,projection_errors=[] ,[]
    for column in range(coefficients.shape[1]):
        harmonic=formal.SphericalHarmonicsCF(input_lmax)
        for value,(ell,m) in zip(coefficients[:,column],formal.spherical_modes(input_lmax)):
            harmonic.sh[ell,m]=complex(value)
        field.Set(harmonic,definedon=mesh.Boundaries('canonical'))
        correction.data=-form.mat*field.vec
        field.vec.data+=inverse*correction
        reaction.data=form.mat*field.vec
        values=reaction.FV().NumPy().copy()
        residuals.append(float(np.linalg.norm(values[free])/max(np.linalg.norm(values),1e-30)))
        projected=np.linalg.solve(gram,conjugate.T@values[boundary])
        result[:,column]=projected
        full_flux=mass_inverse.solve(values[boundary])
        flux_projection=traces[boundary]@projected
        delta=full_flux-flux_projection
        energy=float(np.vdot(full_flux,mb@full_flux).real)
        projection_errors.append(float(np.sqrt(max(np.vdot(delta,mb@delta).real,0)/max(energy,1e-30))))
    meta.update(max_free_dof_relative_residual=max(residuals),seconds=time.perf_counter()-started,
                full_flux_output_projection_errors=projection_errors,
                full_flux_definition='consistent M_bb inverse times independent weak reaction',
                direct_trace_input_lmax=input_lmax,uses_stored_operator=False)
    return result,meta


class RotatedSpec:
    """Duck-typed exact occupancy for a independently remeshed rotated CAD."""
    def __init__(self,base,rotation):
        self.base,self.rotation=base,np.asarray(rotation,dtype=np.float64)
        self.cell_width,self.genus,self.cells,self.id=base.cell_width,base.genus,base.cells,base.id
    def contains(self,points,tolerance=1e-10):
        return self.base.contains(np.asarray(points)@self.rotation,tolerance=tolerance)


def rotated_mesh(spec,profile,rotation):
    # Same registered graded mesh constraints, new CAD and new tetrahedra.
    axis,angle=formal._axis_angle(np.asarray(rotation))
    shape=make_solid(spec)
    if angle>1e-10: shape=shape.Rotate(formal.Axis((0.,0.,0.),tuple(axis)),angle)
    outer=formal.Sphere((0.,0.,0.),1.5)
    for face in outer.faces: face.name='canonical'; face.maxh=profile.canonical_h
    for face in shape.faces: face.name='scatterer'; face.maxh=profile.scatterer_h
    mesh=formal.Mesh(formal.OCCGeometry(outer-shape).GenerateMesh(maxh=profile.background_h,grading=profile.grading))
    mesh.Curve(2)
    return mesh


def rotated_modal_dtn(spec,profile,rotation):
    """Independent assembly via the parent's generic premeshed P2 API.

The parent weak_modal_dtn accepts mesh only indirectly, so perform the same
reaction/consistent Gram algorithm explicitly; never patch global factories.
"""
    from ngsolve import Conj
    started=time.perf_counter()
    mesh=rotated_mesh(spec,profile,rotation)
    rotated=RotatedSpec(spec,rotation)
    meta={**tmatrix._mesh_diagnostics(mesh),'cad_audit':local.audit_cad_mesh(rotated,mesh)}
    if not meta['cad_audit']['passed']: raise RuntimeError('rotated actual CAD audit failed')
    fes=formal.H1(mesh,order=2,complex=True,dirichlet='canonical')
    u,v=fes.TnT()
    a=formal.BilinearForm(fes); a+=formal.grad(u)*formal.grad(v)*formal.dx-4*u*v*formal.dx; a.Assemble()
    inv=a.mat.Inverse(freedofs=fes.FreeDofs())
    mass=formal.BilinearForm(fes); mass+=u*v*formal.ds('canonical'); mass.Assemble()
    boundary=np.flatnonzero(np.asarray(fes.GetDofs(mesh.Boundaries('canonical'))))
    free=np.asarray(fes.FreeDofs(),dtype=bool)
    gf=formal.GridFunction(fes); vector,correction,reaction=gf.vec.CreateVector(),gf.vec.CreateVector(),gf.vec.CreateVector()
    traces=np.empty((fes.ndof,121),complex); test=np.empty((len(boundary),121),complex); mass_trace=np.empty_like(test)
    for j,(ell,m) in enumerate(formal.spherical_modes(10)):
        cf=formal.SphericalHarmonicsCF(ell); cf.sh[ell,m]=1.
        gf.Set(cf,definedon=mesh.Boundaries('canonical')); traces[:,j]=gf.vec.FV().NumPy()
        vector.data=mass.mat*gf.vec; mass_trace[:,j]=vector.FV().NumPy()[boundary]
        gf.Set(Conj(cf),definedon=mesh.Boundaries('canonical')); test[:,j]=gf.vec.FV().NumPy()[boundary]
    gram=test.T@mass_trace; measured=np.empty((121,100),complex); residuals=[]
    for j in range(100):
        gf.vec.FV().NumPy()[:]=traces[:,j]; correction.data=-a.mat*gf.vec; gf.vec.data+=inv*correction
        reaction.data=a.mat*gf.vec; r=reaction.FV().NumPy()
        measured[:,j]=test.T@r[boundary]
        residuals.append(float(np.linalg.norm(r[free])/max(np.linalg.norm(r),1e-30)))
    c=np.linalg.solve(gram,measured)
    meta.update(fes_dofs=fes.ndof,boundary_dofs=len(boundary),mesh_profile=profile.as_dict(),
                mesh_strategy='graded-local-FEM',max_free_dof_relative_residual=max(residuals),
                seconds=time.perf_counter()-started,independent_rotated_remesh=True)
    return c,meta

"""Surface geometry placement and shared prepared-data paths."""
from shared.point_reference_runtime import np
from shared.paths import SURFACE as SIDE, SPLITS
POINTS, K_NEIGHBORS, GRID_SIZE = 2048, 64, 0.02
def placed(points,normals,gids,rotations,context):
    """No S/Z/labels accepted. Context contains only known query conditions."""
    n=len(gids);c=context.reshape(n,128,12)
    r=np.asarray(rotations,np.float32)
    local=np.einsum('opj,okj->opk',points[gids],r)/np.float32(1.5)
    normal=np.einsum('opj,okj->opk',normals[gids],r)
    centers=c[:,0,5:8];direction=c[:,0,8:11]
    pos=local+centers[:,None]
    qpos=c[:,:,2:5]+centers[:,None]
    inc0=c[:,0,0]+1j*c[:,0,1]
    phase=np.sum((local-c[:,0,None,2:5])*direction[:,None,:],axis=-1)*np.float32(3.0)
    inc=inc0[:,None]*np.exp(1j*phase)
    f=np.empty((n,POINTS,15),np.float32)
    f[...,:3]=normal;f[...,3]=inc.real;f[...,4]=inc.imag;f[...,5:8]=local
    f[...,8:11]=centers[:,None,:];f[...,11:14]=direction[:,None,:];f[...,14]=c[:,0,None,11]
    return f.reshape(-1,15),pos.reshape(-1,3),qpos.reshape(-1,3)

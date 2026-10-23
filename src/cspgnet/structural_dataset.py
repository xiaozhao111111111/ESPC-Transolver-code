"""Versioned NumPy-only union reader and exact geometry features for v4.

Legacy sample files and all validation/tests remain immutable. Scene IDs are
global in the union, so old and new sample_00000 never collide.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

RADII=.9*(np.arange(128)+.5)/128


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def radial_occupancy(data):
    if 'geometry_occupancy' in data:
        occupied=np.asarray(data['geometry_occupancy'])
        expected_radii=.9*(np.arange(128,dtype=np.float64)+.5)/128
        if occupied.shape!=(len(data['shape_codes']),128,128):
            raise ValueError('precomputed geometry occupancy must be [objects,128 canonical nodes,128 radii]')
        if not (np.issubdtype(occupied.dtype,np.bool_) or np.issubdtype(occupied.dtype,np.integer)):
            raise ValueError('precomputed geometry occupancy must be a boolean/integer mask')
        if not np.isin(occupied,(0,1)).all(): raise ValueError('geometry occupancy must contain only 0/1')
        if data.get('geometry_occupancy_schema')!='canonical-rays-r0.9-128-midpoints-v1':
            raise ValueError('unknown precomputed geometry occupancy convention')
        if 'geometry_occupancy_radii' not in data or not np.array_equal(
                np.asarray(data['geometry_occupancy_radii'],dtype=np.float64),expected_radii):
            raise ValueError('precomputed occupancy radial coordinates do not match the frozen physical grid')
        return occupied.astype(np.float32,copy=False)
    points=np.asarray(data['interface_points'],float)
    rays=(points/np.linalg.norm(points,axis=1,keepdims=True))[:,None]*RADII[None,:,None]
    codes=np.asarray(data['shape_codes'],int); sizes=np.asarray(data['shape_parameters'],float)
    rotations=np.asarray(data['rotations'],float); result=np.empty((len(codes),128,128),np.float32)
    for i,code in enumerate(codes):
        local=rays@rotations[i]; size=sizes[i]
        if code==0: inside=np.linalg.norm(local,axis=-1)<=size[0]
        elif code==1: inside=np.sum((local/size[:3])**2,axis=-1)<=1
        elif code==2: inside=np.all(np.abs(local)<=size[:3],axis=-1)
        elif code==3:
            r1,r2,d=size[:3]; rho=np.linalg.norm(local[...,:2],axis=-1); z=local[...,2]
            inside=-np.logaddexp(-30*(np.hypot(rho,z+d/2)-r1),-30*(np.hypot(rho,z-d/2)-r2))/30<=0
        elif code==4: inside=np.hypot(np.linalg.norm(local[...,:2],axis=-1)-size[0],local[...,2])<=size[1]
        elif code==5:
            required=('geometry_cells','geometry_cell_count','geometry_cell_width','geometry_lattice_center')
            if any(key not in data for key in required): raise ValueError('structural geometry missing actual cells')
            count=int(data['geometry_cell_count'][i]); width=float(data['geometry_cell_width'][i])
            centers=(np.asarray(data['geometry_cells'][i,:count],float)-data['geometry_lattice_center'][i])*width
            inside=np.zeros((128,128),bool)
            for center in centers: inside|=np.all(np.abs(local-center)<=width/2+1e-10,axis=-1)
        else: raise ValueError(f'unknown shape code {code}; never fall through to torus')
        result[i]=inside
    return result


class UnifiedSceneIndex:
    """A checksummed, scene-indexed reference union (no copying legacy data)."""
    def __init__(self,manifest,verify_files=False):
        self.path=Path(manifest)
        self.manifest=json.loads(self.path.read_text(encoding='utf-8'))
        if self.manifest.get('schema_version') not in ('canonical-response-v4-union','canonical-response-v6-union') or self.manifest.get('status')!='complete':
            raise RuntimeError('union is not complete/released')
        self.items=[]
        physical=set()
        for member in self.manifest['members']:
            path=Path(member['manifest'])
            if not path.is_absolute(): path=self.path.parent/path
            if sha(path)!=member['sha256']: raise RuntimeError('member manifest changed')
            split=json.loads(path.read_text(encoding='utf-8'))
            if split['status']!='complete' or split['split']!='train': raise RuntimeError('union may only contain complete train splits')
            if len(split['samples'])!=member['scenes']: raise RuntimeError('member count mismatch')
            for record in sorted(split['samples'],key=lambda r:r['index']):
                sample=path.parent/record['file']
                identity=record.get('physical_scene_sha256')
                if identity and identity in physical: raise RuntimeError('duplicate physical training scene')
                if identity: physical.add(identity)
                if verify_files and sha(sample)!=record['sha256']: raise RuntimeError('sample corrupted')
                self.items.append((sample,record,member['name']))
        if len(self.items)!=self.manifest['num_samples']: raise RuntimeError('union scene count mismatch')
        self.files=[item[0] for item in self.items]
    def __len__(self): return len(self.items)
    def load(self,index):
        path,record,member=self.items[index]
        with np.load(path,allow_pickle=False) as opened: data={k:opened[k] for k in opened.files}
        return data,{**record,'union_index':index,'source_member':member}


def fit_statistics(index,response_fn):
    """Fit only the union's training scenes, preserving degree-shared RMS."""
    zp=np.zeros(8); sp=np.zeros(8); counts=np.zeros(8,np.int64)
    rs=np.zeros(128); rq=np.zeros(128); rc=0; fp=tp=0.; fc=0; q_energy=[]
    for i in range(len(index)):
        data,_=index.load(i)
        z=np.asarray(data['Z'],np.complex128)
        s=data['S'] if 'S' in data else response_fn(data)
        for ell in range(8):
            sl=slice(ell**2,(ell+1)**2)
            zp[ell]+=np.sum(np.abs(z[...,sl])**2); sp[ell]+=np.sum(np.abs(s[...,sl])**2); counts[ell]+=z[...,sl].size
        radial=radial_occupancy(data).astype(float); rs+=radial.sum(axis=(0,1)); rq+=(radial**2).sum(axis=(0,1)); rc+=radial.shape[0]*128
        total=np.asarray(data['u_total_gt'],np.complex128); incident=np.asarray(data['u_inc'],np.complex128)
        fp+=np.sum(np.abs(total-incident)**2); tp+=np.sum(np.abs(total)**2); fc+=total.size
        q_energy.append(float(np.sum(np.abs(data['q64_gt'])**2)))
        if (i+1)%250==0: print(json.dumps({'training_statistics':i+1,'expected':len(index)}),flush=True)
    mean=rs/rc; variance=np.maximum(rq/rc-mean**2,1e-4)
    union_schema=index.manifest.get('schema_version')
    data_version='canonical-response-v6-structural' if union_schema=='canonical-response-v6-union' else 'canonical-response-v4-structural'
    return {'schema':'canonical-response-v3-train-only-statistics','data_version':data_version,
            'fit_split':'train','fit_samples':len(index),'union_manifest_sha256':sha(index.path),
            'harmonic_degree_complex_rms':np.sqrt(np.maximum(zp/counts,1e-24)).tolist(),
            's_response_degree_complex_rms':np.sqrt(np.maximum(sp/counts,1e-24)).tolist(),
            'radial_occupancy_mean':mean.tolist(),'radial_occupancy_std':np.sqrt(variance).tolist(),
            'radial_variance_floor':1e-4,'scatter_complex_rms':float(np.sqrt(fp/fc)),
            'total_complex_rms':float(np.sqrt(tp/fc)),'q_scene_energy_median':float(np.median(q_energy)),
            'q_loss_stabilizer':float(1e-6*np.median(q_energy)),'canonical_radius':1.5,'no_complex_mean_subtraction':True}

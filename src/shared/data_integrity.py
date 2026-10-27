"""Validate the distributed arrays independently of old experiments or checkpoints."""
import hashlib
import json
from pathlib import Path
from shared.paths import DATA

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()

def verify_data(sizes_only=False):
    manifest_path = DATA/'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('Prepared data is required. Place the data package in '+str(DATA))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for relative, record in manifest['files'].items():
        path = DATA/relative
        if path.is_symlink() or not path.resolve().is_relative_to(DATA) or not path.is_file():
            raise RuntimeError('Missing or unsafe dataset file: '+relative)
        if path.stat().st_size != record['size']:
            raise RuntimeError('Dataset size mismatch: '+relative)
        if not sizes_only and sha(path) != record['sha256']:
            raise RuntimeError('Dataset checksum mismatch: '+relative)
    return {'files': len(manifest['files']), 'sha256_verified': not sizes_only,
            'scene_counts': manifest['scene_counts']}

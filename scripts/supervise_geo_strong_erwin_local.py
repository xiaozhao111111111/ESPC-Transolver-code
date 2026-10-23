"""Single identity-checked GPU queue for the new Geo-strong Erwin experiment."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from point_reference_io_recovery import read_json, write_json
from supervise_point_surface_only_local import identity, terminate_owned_worker, sha

OUT = ROOT/'outputs/erwin_geo_strong_local_20261013'
WORKER = ROOT/'scripts/train_geo_strong_erwin_local.py'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--foreground', action='store_true', required=True)
    parser.parse_args()
    import psutil
    import msvcrt
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'supervisor.lock').open('a+b') as lock:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        if not (read_json(OUT/'geometry/manifest.json') or {}).get('status') == 'complete':
            raise RuntimeError('CPU geometry preparation has not completed')
        transition = read_json(OUT/'transition.json')
        if not transition or not transition.get('old_execution_session_ended'):
            raise RuntimeError('Missing explicit superseded-worker stop evidence')
        # NVIDIA presence is positive evidence even if sandbox process lists omit
        # a worker. Never start while the known old worker still owns GPU memory.
        query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True, timeout=15)
        if str(transition['old_worker_pid']) in query.stdout.split():
            raise RuntimeError('Superseded GPU worker is still present')
        for p in psutil.process_iter(['name', 'cmdline']):
            cmd = p.info['cmdline'] or []
            if p.pid != os.getpid() and (p.info['name'] or '').lower() == 'python.exe' and (
                str(WORKER) in cmd or 'scripts/train_geo_strong_erwin_local.py' in cmd):
                raise RuntimeError('Geo-strong worker already running; refuse duplicate')
        record = {'pid': os.getpid(), 'created_at': psutil.Process().create_time(),
                  'command': psutil.Process().cmdline(), 'source_sha256': sha(Path(__file__)),
                  'status': 'running', 'started_at': time.time()}
        write_json(OUT/'supervisor.json', record)
        try:
            phases = ['qualify', 'train']
            if (read_json(OUT/'erwin_preflight.json') or {}).get('status') == 'passed':
                phases = ['train']  # Training rechecks every source/data binding.
            for phase in phases:
                stamp = time.time_ns()
                logs = OUT/'supervisor'; logs.mkdir(exist_ok=True)
                stdout, stderr = logs/f'{phase}_{stamp}.stdout.log', logs/f'{phase}_{stamp}.stderr.log'
                command = [sys.executable, '-u', str(WORKER), '--'+phase]
                with stdout.open('xb') as out, stderr.open('xb') as err:
                    proc = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err,
                        stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
                worker = identity(proc, command)
                worker.update(phase=phase, stdout=str(stdout.relative_to(ROOT)), stderr=str(stderr.relative_to(ROOT)))
                record.update(phase=phase, worker=worker, updated_at=time.time())
                write_json(logs/f'{phase}_process.json', worker)
                write_json(OUT/'supervisor.json', record)
                print(json.dumps({'event': 'geo_strong_phase_started', **worker}), flush=True)
                try:
                    while proc.poll() is None:
                        p = psutil.Process(proc.pid)
                        if abs(p.create_time()-worker['created_at']) > .1 or p.cmdline() != command:
                            raise RuntimeError('Worker identity mismatch')
                        available = psutil.virtual_memory().available
                        write_json(OUT/'resources.json', {'pid': proc.pid, 'phase': phase,
                            'available_ram_bytes': available, 'rss_bytes': p.memory_info().rss,
                            'updated_at': time.time()})
                        if available < 1024**3:
                            raise RuntimeError('System available RAM below 1 GiB')
                        time.sleep(5)
                except BaseException:
                    terminate_owned_worker(proc, worker)
                    raise
                code = proc.wait()
                write_json(logs/f'{phase}_exit_{stamp}.json', {**worker, 'return_code': code, 'ended_at': time.time()})
                if code:
                    raise RuntimeError(f'{phase} exited {code}; inspect {stderr}')
            record.update(status='complete', completed_at=time.time())
            write_json(OUT/'supervisor.json', record)
        except BaseException as error:
            record.update(status='failed', error=repr(error), traceback=traceback.format_exc(), updated_at=time.time())
            write_json(OUT/'supervisor.json', record)
            raise


if __name__ == '__main__':
    main()

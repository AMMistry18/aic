"""Expand the official AIC UR5e/tool/camera xacro without installing ROS packages."""
import json
import os
from pathlib import Path
import sys
import tempfile
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    for path in Path('/opt/ros/jazzy').glob('**/python3.12/site-packages'):
        sys.path.insert(0, str(path))
    import xacro
    prefix = Path(tempfile.mkdtemp(prefix='aic_xacro_index_'))
    packages = {'aic_assets': ROOT / '.artifacts/upstream-phase1/aic_assets',
                'ur_description': ROOT / '.artifacts/ur_description'}
    for name, source in packages.items():
        index = prefix / 'share/ament_index/resource_index/packages' / name
        index.parent.mkdir(parents=True, exist_ok=True)
        index.touch()
        (prefix / 'share' / name).symlink_to(source, target_is_directory=True)
    os.environ['AMENT_PREFIX_PATH'] = str(prefix) + ':' + os.environ.get('AMENT_PREFIX_PATH', '/opt/ros/jazzy')
    source = ROOT / '.artifacts/upstream-phase1/aic_description/urdf/ur_gz.urdf.xacro'
    document = xacro.process_file(str(source), mappings={
        'name': 'ur5e', 'ur_type': 'ur5e', 'x': '-0.2', 'y': '0.2', 'z': '0',
        'yaw': '-3.141592653589793', 'use_sim_cam': 'true'})
    # The temporary ament index is only for xacro lookup; artifacts must not
    # retain temporary paths (WSL /tmp may disappear between invocations).
    text=document.toxml()
    for name, directory in packages.items():
        text=text.replace(str(prefix/'share'/name), str(directory))
    out = ROOT / '.artifacts/lastinch_resources'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ur5e.urdf').write_text(text)
    env = os.environ.copy()
    models = ROOT / '.artifacts/upstream-phase1/aic_assets/models'
    env['SDF_PATH'] = str(models)
    env['GZ_SIM_RESOURCE_PATH'] = str(models)
    proc = subprocess.run(['gz', 'sdf', '-p', str(models / 'sfp_sc_cable_phase_1/model.sdf')],
                          env=env, capture_output=True, text=True, timeout=30)
    (out / 'cable_expand.log').write_text(proc.stderr)
    if proc.returncode != 0 or '<sdf' not in proc.stdout:
        raise RuntimeError(proc.stderr)
    (out / 'cable.sdf').write_text(proc.stdout)
    (out / 'source.json').write_text(json.dumps({'aic_commit': '7148df9ab62b4b550b101189098f05bc0dba8444',
        'ur_description_tag': '4.3.0', 'ur_description_commit': 'ef93882e17d8aa628837915da4b83208fe5e519a',
        'xacro': str(source)}, indent=2))
    print(out / 'ur5e.urdf')


if __name__ == '__main__':
    main()

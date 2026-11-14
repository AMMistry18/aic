"""Read actual Gazebo Scene service and verify required model instances."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

OUT=Path('/research/.artifacts/kilted_handoff')
result=subprocess.run(['gz','service','-s','/world/aic_lastinch/scene/info',
    '--reqtype','gz.msgs.Empty','--reptype','gz.msgs.Scene','--timeout','10000','--req',''],
    capture_output=True,text=True,timeout=15,check=True)
(OUT/'observed_scene.pbtxt').write_text(result.stdout)
names=re.findall(r'^model\s*\{.*?^  name:\s*"([^"]+)"',result.stdout,re.M|re.S)
required=['task_board','ur5e',*[f'nic_card_mount_{i}' for i in range(5)],
          *[f'sc_port_{i}' for i in range(5)],*[f'cable_{i}' for i in range(5)],
          *[f'{kind}_mount_{i}' for kind in ('sc','sfp') for i in range(5)]]
report={'observed_models':names,'required_models':required,
        'all_required_models_observed':all(name in names for name in required),
        'world_sha256':hashlib.sha256((OUT/'world.sdf').read_bytes()).hexdigest(),
        'scene_service_returncode':result.returncode}
(OUT/'observed_models.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

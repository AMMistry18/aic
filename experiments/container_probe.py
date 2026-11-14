"""Small runtime capability inventory; deliberately excludes credentials/env dumps."""
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys

result={'python':sys.version,'ros_distro':os.environ.get('ROS_DISTRO'),'modules':{},'commands':{}}
for name in ('rclpy','aic_control_interfaces.msg','aic_model','numpy','cv2','torch','ultralytics','pytest'):
    try:
        m=importlib.import_module(name)
        result['modules'][name]={'path':getattr(m,'__file__',None),'version':getattr(m,'__version__',None)}
    except Exception as exc:
        result['modules'][name]={'error':str(exc)}
for name in ('ros2','gz','colcon','pixi'):
    result['commands'][name]=shutil.which(name)
if shutil.which('gz'):
    p=subprocess.run(['gz','sim','--versions'],capture_output=True,text=True,timeout=20)
    result['gazebo_version']=p.stdout.strip()
print(json.dumps(result,indent=2))

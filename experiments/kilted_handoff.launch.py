"""Actual AIC stack with a precompiled robot and prescribed last-inch grasp."""
from pathlib import Path
import json
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ros_gz_bridge.actions import RosGzBridge


def generate_launch_description():
    root=Path('/research/.artifacts/kilted_handoff')
    seed=json.loads((root/'scene.json').read_text())['seed']
    return LaunchDescription([
        Node(package='robot_state_publisher',executable='robot_state_publisher',
             parameters=[{'use_sim_time':True,'robot_description':(root/'robot.urdf').read_text()}]),
        Node(package='aic_adapter',executable='aic_adapter',parameters=[{'use_sim_time':True}]),
        ExecuteProcess(cmd=['gz','sim','--seed',str(seed),'-s','--headless-rendering','-v','3',str(root/'world.sdf')],output='screen'),
        RosGzBridge(bridge_name='research_bridge',
            config_file=str(root/'bridge.yaml')),
        ExecuteProcess(cmd=['python3','/research/experiments/bootstrap_kilted_controller.py'],output='screen'),
    ])

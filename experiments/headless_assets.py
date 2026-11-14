"""Render pinned AIC visual/collision assets with native headless Gazebo.

Run with Linux Python in a machine with Gazebo 8 and its Ogre2 sensor plugins.
This is an asset-integrity scene, not the Kilted robot/evaluator workcell. Meshes,
materials and collisions are copied verbatim; URI resolution, placement, static
staging and removal of model behavior plugins are explicitly recorded.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PIN = "7148df9ab62b4b550b101189098f05bc0dba8444"


def put(parent, tag, value=None, **attrs):
    element = ET.SubElement(parent, tag, attrs)
    if value is not None:
        element.text = str(value)
    return element


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_world(out, drop_probe=False):
    models = ROOT / ".artifacts/upstream-phase1/aic_assets/models"
    root = ET.Element("sdf", version="1.9")
    world = put(root, "world", name="aic_asset_qa")
    physics = put(world, "physics", name="qa_physics", type="ignored")
    put(physics, "max_step_size", .001)
    put(physics, "real_time_factor", 1)
    put(world, "gravity", "0 0 -9.81")
    for library, system in (("physics", "Physics"), ("user-commands", "UserCommands"),
                            ("scene-broadcaster", "SceneBroadcaster"), ("sensors", "Sensors")):
        plugin = put(world, "plugin", filename=f"gz-sim-{library}-system",
                     name=f"gz::sim::systems::{system}")
        if system == "Sensors":
            put(plugin, "render_engine", "ogre2")
    scene = put(world, "scene")
    put(scene, "ambient", ".6 .6 .6 1")
    put(scene, "background", ".18 .20 .23 1")
    light = put(world, "light", name="qa_sun", type="directional")
    put(light, "pose", "0 0 3 0 0 0")
    put(light, "diffuse", ".85 .85 .85 1")
    put(light, "specular", ".2 .2 .2 1")
    put(light, "direction", "-.3 .2 -1")
    put(light, "cast_shadows", "true")
    # SC0 and NIC0 placements are taken from the pinned task_board xacro.
    placements = [
        ("Task Board Base", "board", "0 0 0 0 0 0"),
        ("NIC Card Mount 0", "nic_0", "-.081418 -.0145 .012 0 0 0"),
        ("SC Port 0", "sc_0", "-.075 .0295 .0165 1.57 0 1.57"),
        ("SC Port 1", "sc_1", "-.075 .0705 .0165 1.57 0 1.57"),
        ("SFP Mount", "sfp_mount", ".055 -.10625 .01 0 0 0"),
        ("SC Mount", "sc_mount", ".1 .10625 .012 0 0 0"),
        # Staged standalone plugs are for mesh inspection, not a realistic grasp.
        ("SFP Module", "sfp_module", ".055 -.10625 .045 0 0 0"),
        ("SC Plug", "sc_plug", ".1 .10625 .04 0 0 1.5708"),
    ]
    if drop_probe:
        placements.append(("SC Plug", "qa_drop_sc", ".1 -.03 .15 0 0 0"))
    manifest = []
    for directory, name, pose in placements:
        path = models / directory / "model.sdf"
        original = ET.parse(path).getroot().find("model")
        model = copy.deepcopy(original)
        model.set("name", name)
        for old in model.findall("pose") + model.findall("static"):
            model.remove(old)
        put(model, "pose", pose)
        put(model, "static", "false" if name == "qa_drop_sc" else "true")
        removed = []
        for parent in model.iter():
            for plugin in list(parent.findall("plugin")):
                removed.append(plugin.attrib)
                parent.remove(plugin)
        if name == "qa_drop_sc":
            observer = put(model, "plugin", filename="gz-sim-pose-publisher-system",
                           name="gz::sim::systems::PosePublisher")
            put(observer, "publish_model_pose", "true")
            put(observer, "publish_link_pose", "false")
            put(observer, "use_pose_vector_msg", "true")
            put(observer, "update_frequency", 100)
            put(observer, "topic", "/qa/drop/pose")
        assets = {}
        for uri in model.iter("uri"):
            text = (uri.text or "").strip()
            if "://" in text:
                raise ValueError(f"Unexpected nonlocal URI in pinned model: {text}")
            target = (path.parent / text).resolve()
            if not target.is_file():
                raise FileNotFoundError(target)
            assets[str(target.relative_to(models))] = sha(target)
            uri.text = str(target)
        manifest.append({"source_model": str(path.relative_to(ROOT)), "source_sha256": sha(path),
                         "world_name": name, "pose": pose, "assets_sha256": assets,
                         "visuals": len(list(model.iter("visual"))),
                         "collisions": len(list(model.iter("collision"))),
                         "removed_behavior_plugins": removed,
                         "static_override": name != "qa_drop_sc",
                         "added_pose_observer": name == "qa_drop_sc"})
        world.append(model)
    cameras = [
        ("overview", (.48, -.55, .63), (-.015, 0, .06), .9),
        ("sc_closeup", (-.27, .20, .24), (-.075, .05, .028), .6),
        ("sfp_closeup", (-.29, -.30, .32), (-.07, -.08, .10), .7),
    ]
    for name, position, target, fov in cameras:
        delta = [b - a for a, b in zip(position, target)]
        yaw = math.atan2(delta[1], delta[0])
        pitch = -math.atan2(delta[2], math.hypot(*delta[:2]))
        camera_model = put(world, "model", name=f"camera_{name}")
        put(camera_model, "static", "true")
        put(camera_model, "pose", " ".join(map(str, (*position, 0, pitch, yaw))))
        link = put(camera_model, "link", name="link")
        sensor = put(link, "sensor", name=name, type="camera")
        put(sensor, "always_on", "true")
        put(sensor, "update_rate", 5)
        put(sensor, "topic", f"/qa/{name}/image")
        cam = put(sensor, "camera")
        put(cam, "horizontal_fov", fov)
        image = put(cam, "image")
        put(image, "width", 960)
        put(image, "height", 720)
        put(image, "format", "R8G8B8")
        clip = put(cam, "clip")
        put(clip, "near", .005)
        put(clip, "far", 5)
        capture_dir = out / name
        capture_dir.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    world_path = out / "asset_qa.sdf"
    ET.ElementTree(root).write(world_path, encoding="utf-8", xml_declaration=True)
    return world_path, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / ".artifacts/headless_assets")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--drop-probe", action="store_true", help="Add dynamic original SC plug on the board")
    args = p.parse_args()
    out = args.out.resolve()
    if (out / "run.json").exists():
        p.error("Output contains an existing run; preserve it and choose a fresh --out directory")
    out.mkdir(parents=True, exist_ok=True)
    world, manifest = build_world(out, args.drop_probe)
    (out / "asset_manifest.json").write_text(json.dumps({"upstream_commit": PIN,
        "scope": "asset_integrity_not_full_evaluator", "models": manifest}, indent=2) + "\n")
    if args.prepare_only:
        return
    env = os.environ.copy()
    env.update(GZ_PARTITION=f"aic_asset_qa_{os.getpid()}", EGL_PLATFORM="surfaceless",
               LIBGL_ALWAYS_SOFTWARE="1", QT_QPA_PLATFORM="offscreen")
    # A transport subscription forces rendering. An advertised topic alone does
    # not prove that even one frame was generated.
    os.environ["GZ_PARTITION"] = env["GZ_PARTITION"]
    from gz.transport13 import Node
    from gz.msgs10.image_pb2 import Image as ImageMsg
    from gz.msgs10.scene_pb2 import Scene
    from gz.msgs10.pose_v_pb2 import Pose_V
    from google.protobuf.json_format import MessageToDict
    from PIL import Image
    node = Node()
    frame_metadata = {name: [] for name in ("overview", "sc_closeup", "sfp_closeup")}
    callbacks = []
    lock = threading.Lock()
    target_frames = 12 if args.drop_probe else 3
    drop_poses = []
    def callback_for(name):
        def receive(msg):
            with lock:
                index = len(frame_metadata[name])
                if index >= target_frames:
                    return
                img = Image.frombytes("RGB", (msg.width, msg.height), msg.data,
                                      "raw", "RGB", msg.step)
                image_path = out / name / f"transport_{index:03d}.png"
                img.save(image_path)
                frame_metadata[name].append({"file": str(image_path.relative_to(out)),
                    "width": msg.width, "height": msg.height, "step": msg.step,
                    "pixel_format_type": msg.pixel_format_type,
                    "sim_time_s": msg.header.stamp.sec + msg.header.stamp.nsec / 1e9,
                    "sha256": sha(image_path)})
        return receive
    for name in frame_metadata:
        callback = callback_for(name)
        callbacks.append(callback)
        if not node.subscribe(ImageMsg, f"/qa/{name}/image", callback):
            raise RuntimeError(f"Failed to subscribe to camera {name}")
    def scene_callback(msg):
        (out / "observed_scene.json").write_text(json.dumps(MessageToDict(msg), indent=2) + "\n")
    node.subscribe(Scene, "/world/aic_asset_qa/scene/info", scene_callback)
    def drop_callback(msg):
        for pose in msg.pose:
            if pose.name == "qa_drop_sc":
                drop_poses.append({"sim_time_s": pose.header.stamp.sec + pose.header.stamp.nsec / 1e9,
                                   "pose": MessageToDict(pose)})
    if args.drop_probe:
        node.subscribe(Pose_V, "/qa/drop/pose", drop_callback)
    command = ["gz", "sim", "-s", "-r", "--headless-rendering", "-v", "4", str(world)]
    start = time.monotonic()
    with (out / "gazebo.log").open("wb") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        deadline = start + 90
        while proc.poll() is None and time.monotonic() < deadline:
            if all(len(rows) >= target_frames for rows in frame_metadata.values()):
                break
            time.sleep(.1)
        captured = all(len(rows) >= target_frames for rows in frame_metadata.values())
        if proc.poll() is None:
            proc.send_signal(__import__("signal").SIGINT)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        code = proc.returncode if captured else 124
    captures = {name: [str(f.relative_to(out)) for f in sorted((out / name).glob("*.png"))]
                for name in ("overview", "sc_closeup", "sfp_closeup")}
    result = {"recorded_utc": datetime.now(timezone.utc).isoformat(), "argv": command,
              "returncode": code, "wall_time_s": time.monotonic() - start,
              "renderer": "Ogre2/EGL with software rendering requested",
              "captures": captures, "all_cameras_produced_images": all(captures.values()),
              "transport_frames": frame_metadata,
              "drop_probe_requested": args.drop_probe, "drop_pose_samples": len(drop_poses),
              "full_workcell_validated": False, "insertion_physics_validated": False}
    (out / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.drop_probe:
        (out / "drop_poses.json").write_text(json.dumps(drop_poses, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("captures", "transport_frames")}, indent=2))
    raise SystemExit(0 if code == 0 and captured else 1)


if __name__ == "__main__":
    main()

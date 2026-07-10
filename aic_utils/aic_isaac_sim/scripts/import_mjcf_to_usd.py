#!/usr/bin/env python3
"""Convert the committed AIC MJCF robot/world into Isaac Sim USD assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def _remove_elements(root: ET.Element, tag: str) -> None:
    for parent in root.iter():
        for child in list(parent):
            if child.tag == tag:
                parent.remove(child)


def _make_world_variant(source: Path, destination: Path) -> None:
    """Remove MuJoCo-only plugin/weld declarations before PhysX import."""
    tree = ET.parse(source)
    root = tree.getroot()
    for tag in ("extension", "plugin", "equality"):
        _remove_elements(root, tag)
    # The generated XML lives below the USD output directory. Preserve the
    # source XML's asset resolution instead of making the importer look for all
    # meshes and textures beside that generated file.
    for element in root.iter():
        file_name = element.get("file")
        if file_name and not Path(file_name).is_absolute():
            element.set("file", str((source.parent / file_name).resolve()))
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def _make_robot_variant(source: Path, destination: Path) -> None:
    """Copy robot meshes without legacy OBJ materials rejected by Isaac Sim 6."""
    tree = ET.parse(source)
    mesh_dir = destination.parent / "robot_meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for element in tree.getroot().iter("mesh"):
        file_name = element.get("file")
        if not file_name:
            continue
        original = (source.parent / file_name).resolve()
        if original.suffix.lower() != ".obj":
            continue
        sanitized = mesh_dir / original.name
        text = original.read_text(encoding="utf-8", errors="ignore")
        sanitized.write_text(
            "\n".join(
                line for line in text.splitlines()
                if not line.startswith(("mtllib ", "usemtl "))
            ) + "\n",
            encoding="utf-8",
        )
        element.set("file", str(sanitized))
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _site_pose(source: Path, name: str) -> tuple[list[float], list[float]]:
    root = ET.parse(source).getroot()
    site = root.find(f".//site[@name='{name}']")
    if site is None:
        raise RuntimeError(f"site {name!r} not found in {source}")
    pos = [float(value) for value in site.get("pos", "0 0 0").split()]
    quat = [float(value) for value in site.get("quat", "1 0 0 0").split()]
    if len(pos) != 3 or len(quat) != 4:
        raise RuntimeError(f"invalid pose for site {name!r} in {source}")
    return pos, quat


def _find_relative_prim(
    usd_path: Path, aliases: tuple[str, ...], *, require_rigid: bool = False
) -> str:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"could not open imported USD: {usd_path}")
    default = stage.GetDefaultPrim()
    default_path = str(default.GetPath()) if default and default.IsValid() else ""
    wanted = {_normalized(alias) for alias in aliases}
    matches = []
    for prim in stage.Traverse():
        name = _normalized(prim.GetName())
        path_name = _normalized(str(prim.GetPath()).split("/")[-1])
        if name in wanted or path_name in wanted:
            matches.append(str(prim.GetPath()))
    if not matches:
        raise RuntimeError(f"none of {aliases} found in {usd_path}")
    path = min(matches, key=len)
    if require_rigid:
        prim = stage.GetPrimAtPath(path)
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(
                f"required frame {path} in {usd_path} is not a PhysX rigid body; "
                "the importer likely merged a fixed MJCF body"
            )
    if default_path and path.startswith(default_path):
        path = path[len(default_path):]
    return path or "/"


def _import_asset(source: Path, output_dir: Path, merge_mesh: bool, fix_base: bool) -> Path:
    # Isaac Lab 2.3.x wraps Isaac Sim 5's command-based importer, while newer
    # releases wrap the maintained MJCFImporter API. Using the Lab converter
    # keeps this script compatible with the repository's known-good 2.3.2
    # baseline without reviving deprecated importer calls here.
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    config_kwargs = dict(
        asset_path=str(source),
        usd_dir=str(output_dir),
        usd_file_name=f"{source.stem}.usd",
        force_usd_conversion=True,
        make_instanceable=False,
    )
    # Isaac Lab 3 / Isaac Sim 6 removed ``fix_base`` from MjcfConverterCfg.
    # The committed AIC robot and world already have their bases authored as
    # fixed in the source MJCF, so omitting it preserves the intended result.
    if "fix_base" in getattr(MjcfConverterCfg, "__annotations__", {}):
        config_kwargs["fix_base"] = fix_base
    config = MjcfConverterCfg(**config_kwargs)
    if hasattr(config, "merge_mesh"):
        config.merge_mesh = merge_mesh
    elif merge_mesh:
        raise RuntimeError("--merge-mesh is not supported by this Isaac Lab MJCF converter")
    result = Path(MjcfConverter(config).usd_path).resolve()
    # Isaac Sim 6 schedules MJCF file emission on Kit's update loop.  Advance
    # it explicitly when running headless so import completion is observable.
    import omni.kit.app
    for _ in range(30):
        omni.kit.app.get_app().update()
    # Sim 6's MJCFImporter authors the converted prims into the active stage
    # but does not always persist the stage when driven through Lab's wrapper.
    # Persist that authored stage explicitly for headless conversion.
    if not result.is_file():
        import omni.usd
        result.parent.mkdir(parents=True, exist_ok=True)
        if not omni.usd.get_context().save_as_stage(str(result)):
            raise RuntimeError(f"Isaac Sim did not save imported stage to {result}")
    if not result.is_file():
        raise RuntimeError(f"MJCF importer did not produce a USD file: {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", type=Path, required=True, help="top-level scene.xml")
    parser.add_argument("--usd-dir", type=Path, required=True, help="generated asset directory")
    parser.add_argument("--merge-mesh", action="store_true")
    args = parser.parse_args()

    scene = args.mjcf.expanduser().resolve()
    if not scene.is_file() or scene.name != "scene.xml":
        parser.error("--mjcf must point to the committed top-level scene.xml")
    source_dir = scene.parent
    robot_xml = source_dir / "aic_robot.xml"
    world_xml = source_dir / "aic_world.xml"
    if not robot_xml.is_file() or not world_xml.is_file():
        parser.error("scene.xml must be beside aic_robot.xml and aic_world.xml")

    output_dir = args.usd_dir.expanduser().resolve()
    intermediate = output_dir / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    robot_variant = intermediate / "aic_robot_isaac.xml"
    world_variant = intermediate / "aic_world_isaac.xml"
    _make_robot_variant(robot_xml, robot_variant)
    _make_world_variant(world_xml, world_variant)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        robot_usd = _import_asset(robot_variant, output_dir, args.merge_mesh, fix_base=True)
        world_usd = _import_asset(world_variant, output_dir, args.merge_mesh, fix_base=True)
        tcp_pos, tcp_quat = _site_pose(robot_xml, "gripper_tcp")
        manifest = {
            "schema_version": 1,
            "source_scene": str(scene),
            "source_sha256": hashlib.sha256(
                robot_xml.read_bytes() + world_xml.read_bytes() + scene.read_bytes()
            ).hexdigest(),
            "robot_usd": str(robot_usd),
            "world_usd": str(world_usd),
            "robot_tool_relpath": _find_relative_prim(
                robot_usd,
                ("ati/tool_link", "ati_tool_link", "tool_link"),
                require_rigid=True,
            ),
            "robot_tcp_relpath": _find_relative_prim(
                robot_usd, ("gripper_tcp", "gripper/tcp", "tcp")
            ),
            "robot_tcp_offset_pos": tcp_pos,
            "robot_tcp_offset_quat_wxyz": tcp_quat,
            "world_plug_relpath": _find_relative_prim(
                world_usd, ("lc_plug_link", "lc_plug"), require_rigid=True
            ),
            "world_tip_relpath": _find_relative_prim(
                world_usd, ("sfp_tip_link", "sfp_tip"), require_rigid=True
            ),
            "world_tail_relpath": _find_relative_prim(
                world_usd, ("sfp_module_link", "sfp_module"), require_rigid=True
            ),
            "world_target_relpath": _find_relative_prim(
                world_usd, ("sfp_port_1_link_entrance",), require_rigid=True
            ),
            "removed_mujoco_features": [
                "mujoco.elasticity.cable plugin",
                "MJCF equality weld (re-authored as a USD fixed joint)",
            ],
        }
        manifest_path = output_dir / "asset_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"robot USD: {robot_usd}")
        print(f"world USD: {world_usd}")
        print(f"asset manifest: {manifest_path}")
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

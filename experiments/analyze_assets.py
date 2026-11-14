"""Verify rendered mesh provenance and analyse the controlled SC drop probe."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rpy(values):
    x, y, z = values
    cx, cy, cz = np.cos([x, y, z])
    sx, sy, sz = np.sin([x, y, z])
    return np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                     [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx], [-sy, cy*sx, cy*cx]])


def pose6(element):
    vals = list(map(float, element.findtext("pose", "0 0 0 0 0 0").split()))
    return np.array(vals[:3]), rpy(vals[3:])


def collision_bottom(model, pos, rotation):
    bounds = []
    for link in model.findall("link"):
        lp, lr = pose6(link)
        for collision in link.findall("collision"):
            cp, cr = pose6(collision)
            center = pos + rotation @ (lp + lr @ cp)
            R = rotation @ lr @ cr
            geometry = collision.find("geometry")
            box, cylinder = geometry.find("box"), geometry.find("cylinder")
            if box is not None:
                size = np.array(list(map(float, box.findtext("size").split())))
                support = np.abs(R[2]) @ (size / 2)
            elif cylinder is not None:
                support = (float(cylinder.findtext("radius")) * np.linalg.norm(R[2, :2])
                           + float(cylinder.findtext("length")) / 2 * abs(R[2, 2]))
            else:
                raise ValueError("Drop bound needs analytic box/cylinder collision geometry")
            bounds.append(center[2] - support)
    return min(bounds)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / ".artifacts/headless_assets_drop")
    p.add_argument("--out", type=Path, default=ROOT / "experiments/evidence/headless_assets")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.run / "asset_manifest.json").read_text())
    run = json.loads((args.run / "run.json").read_text())
    scene = json.loads((args.run / "observed_scene.json").read_text())
    world = ET.parse(args.run / "asset_qa.sdf").getroot().find("world")
    observed = {m["name"]: m for m in scene["model"]}
    checked = []
    for row in manifest["models"]:
        source = ROOT / row["source_model"]
        assert sha(source) == row["source_sha256"], "Source model changed"
        original = ET.parse(source).getroot().find("model")
        generated = world.find(f"model[@name='{row['world_name']}']")
        # Collision tags include poses, dimensions and surfaces; no tolerance.
        canon = lambda x: ET.canonicalize(ET.tostring(x, encoding="unicode"), strip_text=True)
        # Behavior plugins also contain <collision>name</collision> references.
        # Compare actual definitions, which have a name attribute.
        definitions = lambda model: [canon(x) for x in model.iter("collision") if "name" in x.attrib]
        assert definitions(original) == definitions(generated)
        seen = observed[row["world_name"]]
        visuals = [v for link in seen.get("link", []) for v in link.get("visual", [])]
        assert len(visuals) == row["visuals"], "Gazebo observed visual count differs from source"
        for visual in visuals:
            mesh = visual.get("geometry", {}).get("mesh")
            if mesh:
                name = Path(mesh["filename"]).name
                matches = [h for path, h in row["assets_sha256"].items() if Path(path).name == name]
                assert len(matches) == 1, f"Unexpected observed mesh: {name}"
                source_relative = [path for path in row["assets_sha256"] if Path(path).name == name][0]
                expected_suffix = "/aic_assets/models/" + source_relative.replace("\\", "/")
                assert mesh["filename"].replace("\\", "/").endswith(expected_suffix), "Observed mesh comes from a different asset directory"
                assert all(float(mesh["scale"][axis]) == 1 for axis in ("x", "y", "z"))
                assert sha(source.parent / name) == matches[0], "Mesh hash differs"
        checked.append({"model": row["world_name"], "visuals": len(visuals),
                        "collisions_preserved": row["collisions"], "mesh_hashes_match": True})
    cameras = {}
    for name, frames in run["transport_frames"].items():
        assert len(frames) >= 3
        first, last = (args.run / frames[i]["file"] for i in (0, -1))
        im = np.asarray(Image.open(last))
        assert im.shape == (720, 960, 3) and np.std(im) > 10, "Blank or invalid camera frame"
        assert sha(last) == frames[-1]["sha256"]
        shutil.copy2(last, args.out / f"{name}.png")
        if name == "overview":
            shutil.copy2(first, args.out / "overview_initial.png")
        cameras[name] = {"frames": len(frames), "width": 960, "height": 720,
                         "sim_time_first_s": frames[0]["sim_time_s"],
                         "sim_time_last_s": frames[-1]["sim_time_s"],
                         "first_last_pixels_changed": int(np.any(np.asarray(Image.open(first)) != im, axis=2).sum()),
                         "last_sha256": sha(last)}
    result = {"scope": "pinned_asset_scene_not_robot_evaluator", "upstream_commit": manifest["upstream_commit"],
              "observed_models": checked, "cameras": cameras,
              "headless_command": run["argv"], "renderer": run["renderer"],
              "full_workcell_validated": False, "insertion_physics_validated": False}
    poses = json.loads((args.run / "drop_poses.json").read_text())
    plug = ET.parse(ROOT / ".artifacts/upstream-phase1/aic_assets/models/SC Plug/model.sdf").getroot().find("model")
    board = ET.parse(ROOT / ".artifacts/upstream-phase1/aic_assets/models/Task Board Base/model.sdf")
    base = board.find(".//collision[@name='v1015083001_collider_box']")
    board_top = pose6(base)[0][2] + float(base.findtext("geometry/box/size").split()[2]) / 2
    trajectory = []
    for row in poses:
        pose = row["pose"]
        stamp = pose["header"]["stamp"]
        t = int(stamp.get("sec", 0)) + int(stamp.get("nsec", 0)) / 1e9
        pos = np.array([pose["position"].get(k, 0) for k in ("x", "y", "z")])
        q = pose["orientation"]
        x, y, z, w = [q.get(k, 0) for k in ("x", "y", "z", "w")]
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        trajectory.append([t, pos[2], collision_bottom(plug, pos, R) - board_top])
    trajectory = np.asarray(trajectory)
    assert len(trajectory) > 100 and np.all(np.diff(trajectory[:, 0]) > 0)
    tail = trajectory[trajectory[:, 0] >= trajectory[-1, 0] - .3]
    result["drop_probe"] = {
        "samples": len(trajectory), "board_collision_top_m": board_top,
        "first_model_z_m": float(trajectory[0, 1]), "last_model_z_m": float(trajectory[-1, 1]),
        "final_collision_clearance_mm": float(trajectory[-1, 2] * 1000),
        "peak_sampled_geometric_penetration_mm": float(max(0, -trajectory[:, 2].min()) * 1000),
        "terminal_0_3s_model_z_span_mm": float(np.ptp(tail[:, 1]) * 1000),
        "fell_and_settled_at_board": bool(trajectory[0, 1] - trajectory[-1, 1] > .1
            and abs(trajectory[-1, 2]) < .0002 and np.ptp(tail[:, 1]) < .0002),
        "interpretation": "Single gravity/drop sanity check; impact penetration is not insertion accuracy",
    }
    np.savetxt(args.out / "drop_trajectory.csv", trajectory, delimiter=",",
               header="sim_time_s,model_z_m,collision_clearance_to_board_m", comments="")
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), layout="constrained")
    axes[0].plot(trajectory[:, 0], trajectory[:, 1] * 1000)
    axes[0].set(xlabel="Simulated time (s)", ylabel="SC model height (mm)")
    axes[1].plot(trajectory[:, 0], trajectory[:, 2] * 1000)
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set(xlabel="Simulated time (s)", ylabel="Collision clearance to board (mm)", ylim=(-3, 2))
    for ax in axes:
        ax.grid(alpha=.2)
    for ext in ("png", "svg"):
        fig.savefig(args.out / f"drop_probe.{ext}", dpi=250)
    plt.close(fig)
    (args.out / "integrity.json").write_text(json.dumps(result, indent=2) + "\n")
    for name in ("asset_manifest.json", "run.json", "observed_scene.json", "drop_poses.json", "asset_qa.sdf"):
        shutil.copy2(args.run / name, args.out / name)
    print(json.dumps(result["drop_probe"], indent=2))
    print(f"Verified {len(checked)} observed models and three RGB cameras: {args.out}")


if __name__ == "__main__":
    main()

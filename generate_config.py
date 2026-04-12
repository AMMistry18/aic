#!/usr/bin/env python3
import argparse, random, yaml, os

NIC_RAIL_TRANSLATION_RANGE   = (-0.0215, 0.0234)
SC_RAIL_TRANSLATION_RANGE    = (-0.06,   0.055)
MOUNT_RAIL_TRANSLATION_RANGE = (-0.09425, 0.09425)
BOARD_X_RANGE    = (0.13, 0.20)
BOARD_Y_RANGE    = (-0.25, 0.10)
BOARD_Z          = 1.14
BOARD_YAW_BANDS  = [(2.80,2.95),(2.95,3.05),(3.05,3.20),(3.20,3.35)]
GRIPPER_Z_RANGE  = (0.038, 0.048)
GRIPPER_ROLL, GRIPPER_PITCH, GRIPPER_YAW = 0.4432, -0.4838, 1.3303

def rand(lo, hi): return round(random.uniform(lo, hi), 4)
def rand_t(r):    return rand(*r)

def build_nic_rails(active_rails, trial_idx):
    """
    Build NIC rail spawn dict.

    active_rails: iterable of rail indices (0..4) to spawn NIC cards on.
    """
    active = set(active_rails)
    out = {}
    for i in range(5):
        if i in active:
            # entity_name must be unique per spawned NIC to avoid Gazebo name collisions
            entity = f"nic_card_{trial_idx}_{i}"
            out[f"nic_rail_{i}"] = {
                "entity_present": True,
                "entity_name": entity,
                "entity_pose": {
                    "translation": rand_t(NIC_RAIL_TRANSLATION_RANGE),
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": round(random.uniform(-0.1, 0.1), 4),
                },
            }
        else:
            out[f"nic_rail_{i}"] = {"entity_present": False}
    return out

def build_sc_rails(active_rails, trial_idx):
    """
    Build SC rail spawn dict.

    active_rails: iterable of rail indices (0..1) to spawn SC ports on.
    """
    active = set(active_rails)
    out = {}
    for i in range(2):
        if i in active:
            entity = f"sc_mount_{trial_idx}_{i}"
            out[f"sc_rail_{i}"] = {
                "entity_present": True,
                "entity_name": entity,
                "entity_pose": {
                    "translation": rand_t(SC_RAIL_TRANSLATION_RANGE),
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": round(random.uniform(-0.15, 0.15), 4),
                },
            }
        else:
            out[f"sc_rail_{i}"] = {"entity_present": False}
    return out

def choose_nic_spawn_rails(trial_idx, num_trials, mode: str):
    """
    mode:
      - 'one': always 1 NIC
      - 'mixed': exactly half the trials spawn 1 NIC, half spawn 2–5 NICs
        (parity on trial_idx). Multi-card count is weighted toward 2, then 3, 4, 5.
      - 'all': always 5 NICs
    """
    if mode == "all":
        return list(range(5))
    if mode == "one":
        return [random.randrange(5)]

    # mixed — strict 50/50 single vs multi (stable via trial index parity).
    if (trial_idx % 2) == 0:
        return [random.randrange(5)]

    count = random.choices(
        population=[2, 3, 4, 5],
        weights=[0.38, 0.30, 0.20, 0.12],
        k=1,
    )[0]
    return random.sample(list(range(5)), k=count)

def choose_sc_spawn_rails(mode: str):
    """
    mode:
      - 'one': spawn exactly 1 SC port (random rail)
      - 'both': spawn both SC ports
      - 'mixed': spawn one or both (50/50)
    """
    if mode == "both":
        return [0, 1]
    if mode == "one":
        return [random.randrange(2)]
    # mixed
    return [0, 1] if random.random() < 0.5 else [random.randrange(2)]

def build_mount_rails(trial_idx):
    """
    6 mount rails. Each gets a unique entity_name scoped to this trial
    so no two spawned entities in the same trial share a name.
    """
    def maybe(prefix, slot):
        if random.choice([True, False]):
            return {
                "entity_present": True,
                "entity_name": f"{prefix}_{trial_idx}_{slot}",
                "entity_pose": {"translation": rand_t(MOUNT_RAIL_TRANSLATION_RANGE),
                                "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
            }
        return {"entity_present": False}

    return {
        "lc_mount_rail_0":  maybe("lc_mount",  0),
        "sfp_mount_rail_0": maybe("sfp_mount", 0),
        "sc_mount_rail_0":  maybe("sc_mount",  0),
        "lc_mount_rail_1":  maybe("lc_mount",  1),
        "sfp_mount_rail_1": maybe("sfp_mount", 1),
        "sc_mount_rail_1":  maybe("sc_mount",  1),
    }

def build_trial(trial_idx, plug_type, nic_rail, sc_rail, yaw_band):
    board = {"x": rand(*BOARD_X_RANGE), "y": rand(*BOARD_Y_RANGE), "z": BOARD_Z,
             "roll": 0.0, "pitch": 0.0, "yaw": rand(*yaw_band)}
    cable_name = f"cable_{trial_idx}"

    if plug_type == "sfp":
        nic_rails  = build_nic_rails(nic_rail, trial_idx)
        sc_rails   = build_sc_rails(sc_rail, trial_idx)
        cable_type = "sfp_sc_cable"
        task = {"cable_type": "sfp_sc", "cable_name": cable_name,
                "plug_type": "sfp", "plug_name": "sfp_tip",
                "port_type": "sfp", "port_name": "sfp_port_0",
                "target_module_name": f"nic_card_mount_{nic_rail}",
                "time_limit": 180}
    else:
        nic_rails  = {f"nic_rail_{i}": {"entity_present": False} for i in range(5)}
        sc_rails   = build_sc_rails(sc_rail, trial_idx)
        cable_type = "sfp_sc_cable_reversed"
        task = {"cable_type": "sfp_sc", "cable_name": cable_name,
                "plug_type": "sc", "plug_name": "sc_tip",
                "port_type": "sc", "port_name": "sc_port_base",
                "target_module_name": f"sc_port_{sc_rail}",
                "time_limit": 180}

    cable = {cable_name: {
        "pose": {"gripper_offset": {"x": 0.0, "y": 0.015385,
                                    "z": rand(*GRIPPER_Z_RANGE)},
                 "roll": GRIPPER_ROLL, "pitch": GRIPPER_PITCH,
                 "yaw": GRIPPER_YAW},
        "attach_cable_to_gripper": True,
        "cable_type": cable_type
    }}

    tb = {"pose": board}
    tb.update(nic_rails)
    tb.update(sc_rails)
    tb.update(build_mount_rails(trial_idx))

    return {"scene": {"task_board": tb, "cables": cable},
            "tasks": {"task_1": task}}

def build_trial(trial_idx, plug_type, nic_spawn_rails, sc_spawn_rails, yaw_band):
    board = {"x": rand(*BOARD_X_RANGE), "y": rand(*BOARD_Y_RANGE), "z": BOARD_Z,
             "roll": 0.0, "pitch": 0.0, "yaw": rand(*yaw_band)}
    cable_name = f"cable_{trial_idx}"

    nic_rails = build_nic_rails(nic_spawn_rails, trial_idx)
    sc_rails = build_sc_rails(sc_spawn_rails, trial_idx)

    if plug_type == "sfp":
        cable_type = "sfp_sc_cable"
        target_rail = random.choice(list(nic_spawn_rails))
        target_port = random.choice(["sfp_port_0", "sfp_port_1"])
        task = {
            "cable_type": "sfp_sc",
            "cable_name": cable_name,
            "plug_type": "sfp",
            "plug_name": "sfp_tip",
            "port_type": "sfp",
            "port_name": target_port,
            "target_module_name": f"nic_card_mount_{target_rail}",
            "time_limit": 180,
        }
    else:
        cable_type = "sfp_sc_cable_reversed"
        # Only one SC port is the target port, even if both are spawned
        target_sc = random.choice(list(sc_spawn_rails))
        task = {
            "cable_type": "sfp_sc",
            "cable_name": cable_name,
            "plug_type": "sc",
            "plug_name": "sc_tip",
            "port_type": "sc",
            "port_name": "sc_port_base",
            "target_module_name": f"sc_port_{target_sc}",
            "time_limit": 180,
        }

    cable = {cable_name: {
        "pose": {"gripper_offset": {"x": 0.0, "y": 0.015385,
                                    "z": rand(*GRIPPER_Z_RANGE)},
                 "roll": GRIPPER_ROLL, "pitch": GRIPPER_PITCH,
                 "yaw": GRIPPER_YAW},
        "attach_cable_to_gripper": True,
        "cable_type": cable_type
    }}

    tb = {"pose": board}
    tb.update(nic_rails)
    tb.update(sc_rails)
    tb.update(build_mount_rails(trial_idx))

    return {"scene": {"task_board": tb, "cables": cable},
            "tasks": {"task_1": task}}


def build_config(
    num_trials,
    seed=None,
    task_mode: str = "mixed",
    nic_spawn_mode: str = "one",
    sc_spawn_mode: str = "one",
):
    if seed is not None:
        random.seed(seed)

    # Stratified pools — guarantee intra-run diversity for yaw bands
    yaw_pool = BOARD_YAW_BANDS * ((num_trials // len(BOARD_YAW_BANDS)) + 1)
    random.shuffle(yaw_pool)

    if task_mode == "nic_only":
        plugs = ["sfp"] * num_trials
    elif task_mode == "sc_only":
        plugs = ["sc"] * num_trials
    else:
        # 2/3 sfp, 1/3 sc for class balance
        plugs = ["sc" if i % 3 == 2 else "sfp" for i in range(num_trials)]
        random.shuffle(plugs)

    scoring_topics = [
        {"topic": {"name": "/joint_states",              "type": "sensor_msgs/msg/JointState"}},
        {"topic": {"name": "/tf",                        "type": "tf2_msgs/msg/TFMessage"}},
        {"topic": {"name": "/tf_static",                 "type": "tf2_msgs/msg/TFMessage", "latched": True}},
        {"topic": {"name": "/scoring/tf",                "type": "tf2_msgs/msg/TFMessage"}},
        {"topic": {"name": "/aic/gazebo/contacts/off_limit", "type": "ros_gz_interfaces/msg/Contacts"}},
        {"topic": {"name": "/fts_broadcaster/wrench",    "type": "geometry_msgs/msg/WrenchStamped"}},
        {"topic": {"name": "/aic_controller/joint_commands", "type": "aic_control_interfaces/msg/JointMotionUpdate"}},
        {"topic": {"name": "/aic_controller/pose_commands",  "type": "aic_control_interfaces/msg/MotionUpdate"}},
        {"topic": {"name": "/scoring/insertion_event",   "type": "std_msgs/msg/String"}},
        {"topic": {"name": "/aic_controller/controller_state", "type": "aic_control_interfaces/msg/ControllerState"}},
    ]

    config = {
        "scoring": {"topics": scoring_topics},
        "task_board_limits": {
            "nic_rail":   {"min_translation": -0.0215,  "max_translation": 0.0234},
            "sc_rail":    {"min_translation": -0.06,    "max_translation": 0.055},
            "mount_rail": {"min_translation": -0.09425, "max_translation": 0.09425},
        },
        "trials": {},
        "robot": {"home_joint_positions": {
            "shoulder_pan_joint":  -0.1597,
            "shoulder_lift_joint": -1.3542,
            "elbow_joint":         -1.6648,
            "wrist_1_joint":       -1.6933,
            "wrist_2_joint":        1.5710,
            "wrist_3_joint":        1.4110,
        }}
    }

    for i in range(num_trials):
        # Choose spawns independently of task type (so you can spawn SC ports but run NIC-only tasks).
        nic_spawn_rails = choose_nic_spawn_rails(i, num_trials, nic_spawn_mode)
        sc_spawn_rails = choose_sc_spawn_rails(sc_spawn_mode)

        # If we're running SC-only tasks, NIC spawns are irrelevant but harmless.
        # If we're running NIC-only tasks, SC spawns can still be present for realism.
        config["trials"][f"trial_{i+1}"] = build_trial(
            i, plugs[i], nic_spawn_rails, sc_spawn_rails, yaw_pool[i])

    return config

def main():
    parser = argparse.ArgumentParser(
        description="Generate randomized AIC sample_config.yaml")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--output", type=str, default=os.path.expanduser(
        "~/ws_aic/src/aic/aic_engine/config/sample_config.yaml"))
    parser.add_argument("--seed",    type=int,  default=None)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--task-mode",
        choices=["mixed", "nic_only", "sc_only"],
        default="mixed",
        help="Which insertion tasks to generate (spawns are independent).",
    )
    parser.add_argument(
        "--nic-spawn-mode",
        choices=["one", "mixed", "all"],
        default="one",
        help="How many NIC cards to spawn per trial.",
    )
    parser.add_argument(
        "--sc-spawn-mode",
        choices=["one", "both", "mixed"],
        default="one",
        help="How many SC ports to spawn per trial.",
    )
    args = parser.parse_args()

    config = build_config(
        args.trials,
        args.seed,
        task_mode=args.task_mode,
        nic_spawn_mode=args.nic_spawn_mode,
        sc_spawn_mode=args.sc_spawn_mode,
    )

    if args.preview:
        print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    else:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"[generate_config] Wrote {args.trials} trials → {args.output}")
        for k, v in config["trials"].items():
            t = v["tasks"]["task_1"]
            b = v["scene"]["task_board"]["pose"]
            tb = v["scene"]["task_board"]
            nic_n = sum(1 for i in range(5) if tb.get(f"nic_rail_{i}", {}).get("entity_present"))
            sc_n = sum(1 for i in range(2) if tb.get(f"sc_rail_{i}", {}).get("entity_present"))
            print(f"  {k}: plug={t['plug_type']:3s}  "
                  f"target={t['target_module_name']:20s}  "
                  f"nic={nic_n} sc={sc_n}  yaw={b['yaw']:.3f}  y={b['y']:.3f}")

if __name__ == "__main__":
    main()
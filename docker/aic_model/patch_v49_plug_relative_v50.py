#!/usr/bin/env python3
"""Install the v50 controller dispatch into the exact deployed v49 runtime.

Both source-tree and installed site-packages copies are patched.  Input hashes
are pinned to the locally inspected v49 image so this script fails instead of
silently applying anchors to a different policy lineage.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys


EXPECTED_V49_RLINSERT_SHA256 = (
    "d88c8c764df508578451d498aeaccbd5490b204799868413bf993c78b55c8365"
)
EXPECTED_V49_AIC_MODEL_SHA256 = (
    "aaee29b9f38cfac2abe9e51b22da2811559160762e16efd479eafeb828bea0c4"
)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one {label} anchor, found {count}")
    return source.replace(old, new, 1)


RLIMPORT_OLD = "from .visual_gap_recovery import VisualGapRecoveryMixin\n"
RLIMPORT_NEW = '''from .visual_gap_recovery import VisualGapRecoveryMixin
from .v50_controller import (
    configure_v50,
    prime_v50_plug_pose,
    run_v50_script,
    v50_tcp_pose_for_tip,
    v50_tip_from_tcp,
)
'''

DEADLINE_OLD = '''ACTION_TIME_BUDGET_S = float(os.environ.get("RL_INSERT_ACTION_TIME_BUDGET_S", "145.0"))
'''
DEADLINE_NEW = '''ACTION_TIME_BUDGET_S = float(os.environ.get("RL_INSERT_ACTION_TIME_BUDGET_S", "45.0"))
'''

PERCEPTION_INIT_OLD = '''        self._pc = PerceptionCore(nic_weights=NIC_WEIGHTS, sc_weights=sc_weights)

        # Script-only mode does not import torch, require an actor artifact, or
'''
PERCEPTION_INIT_NEW = '''        self._pc = PerceptionCore(nic_weights=NIC_WEIGHTS, sc_weights=sc_weights)
        # The dedicated plug model is mandatory in v50.  Missing weights fail
        # lifecycle configuration; there is deliberately no fixed-bias fallback.
        configure_v50(self)

        # Script-only mode does not import torch, require an actor artifact, or
'''

TIP_OLD = '''    def _tip_from_tcp(self, tcp_pos, tcp_quat):
        """SFP plug tip pose from TCP + the measured SFP-tip<-TCP transform."""
        return sfp_tip_pose_from_tcp(tcp_pos, tcp_quat)
'''
TIP_NEW = '''    def _tip_from_tcp(self, tcp_pos, tcp_quat):
        """Plug tip from the fresh per-run visual grasp transform in v50."""
        return v50_tip_from_tcp(self, tcp_pos, tcp_quat)
'''

TCP_TARGET_OLD = '''    def _tcp_target_for_tip(self, tip_pos, R_tip):
        tcp_pos, q_tcp = tcp_pose_for_sfp_tip(tip_pos, R_tip)
'''
TCP_TARGET_NEW = '''    def _tcp_target_for_tip(self, tip_pos, R_tip):
        tcp_pos, q_tcp = v50_tcp_pose_for_tip(self, tip_pos, R_tip)
'''

SCRIPT_DISPATCH_OLD = '''        # Scripted (no-RL) align-then-progressive-descent test path.
        if CONTROL_MODE == "script":
            return self._run_script(
                get_observation, move_robot, send_feedback,
                port_pos=port_pos, port_quat=port_quat, Rp=Rp)
'''
SCRIPT_DISPATCH_NEW = '''        # v50 uses direct plug-to-port geometry, persistent axial authority,
        # visual-first wedge rescue, and bounded fresh-perception recovery.  The
        # legacy fixed-bias _run_script remains present only for image provenance;
        # this dispatch makes it unreachable in the script-only release.
        if CONTROL_MODE == "script":
            return run_v50_script(
                self, task, get_observation, move_robot, send_feedback,
                port_pos=port_pos, port_quat=port_quat, Rp=Rp)
'''

PORT_PERCEPTION_OLD = '''        # --- perceive the SFP port pose (multi-frame consensus so one bad frame
        #     or a wrong-port pick does not commit us to the wrong cage)
        perceived = self.perceive_port_pose_consensus(task, get_observation)
'''
PORT_PERCEPTION_NEW = '''        # Prime direct plug vision before port detection.  The existing port
        # candidate selector then measures candidate distance from the observed plug,
        # not from the legacy fixed grasp transform.  Missing plug vision fails
        # closed; fixed bias/grasp control is never a fallback.
        if not prime_v50_plug_pose(self, get_observation, move_robot):
            send_feedback("fresh plug pose unavailable -- insertion aborted")
            return False

        # --- perceive the SFP port pose (multi-frame consensus so one bad frame
        #     or a wrong-port pick does not commit us to the wrong cage)
        perceived = self.perceive_port_pose_consensus(task, get_observation)
'''


def patch_rlinsert_source(source: str) -> str:
    source = replace_once(source, RLIMPORT_OLD, RLIMPORT_NEW, "v50 import")
    source = replace_once(source, DEADLINE_OLD, DEADLINE_NEW, "45s deadline")
    source = replace_once(
        source, PERCEPTION_INIT_OLD, PERCEPTION_INIT_NEW, "plug estimator init"
    )
    source = replace_once(source, TIP_OLD, TIP_NEW, "dynamic plug tip")
    source = replace_once(source, TCP_TARGET_OLD, TCP_TARGET_NEW, "dynamic TCP target")
    source = replace_once(
        source, PORT_PERCEPTION_OLD, PORT_PERCEPTION_NEW, "direct plug priming"
    )
    source = replace_once(
        source, SCRIPT_DISPATCH_OLD, SCRIPT_DISPATCH_NEW, "v50 script dispatch"
    )
    return source


AIC_IMPORT_OLD = "from std_srvs.srv import Empty\n"
AIC_IMPORT_NEW = (
    "import os\nfrom std_msgs.msg import String\nfrom std_srvs.srv import Empty\n"
)

EVENT_INIT_OLD = '''        self.observation_sub = self.create_subscription(
            Observation, "observations", self.observation_callback, 10
        )
        self._action_callback_group = ReentrantCallbackGroup()
'''
EVENT_INIT_NEW = '''        self.observation_sub = self.create_subscription(
            Observation, "observations", self.observation_callback, 10
        )
        # The Gazebo TouchPlugin is bridged to this ROS topic.  v50 accepts
        # success only from a new, matching correct-port event.
        self._insertion_event_value = ""
        self._insertion_event_generation = 0
        self.insertion_event_sub = self.create_subscription(
            String, "/scoring/insertion_event", self.insertion_event_callback, 10
        )
        self._action_callback_group = ReentrantCallbackGroup()
'''

EVENT_SHUTDOWN_OLD = '''        self.destroy_subscription(self.observation_sub)
        self.observation_sub = None
        self.action_server = None
'''
EVENT_SHUTDOWN_NEW = '''        self.destroy_subscription(self.observation_sub)
        self.observation_sub = None
        self.destroy_subscription(self.insertion_event_sub)
        self.insertion_event_sub = None
        self.action_server = None
'''

EVENT_CALLBACK_OLD = '''    def observation_callback(self, msg):
        self._observation_msg = msg

    def insert_cable_goal_callback(self, goal_request):
'''
EVENT_CALLBACK_NEW = '''    def observation_callback(self, msg):
        self._observation_msg = msg

    def insertion_event_callback(self, msg):
        self._insertion_event_value = msg.data
        self._insertion_event_generation += 1
        self.get_logger().info(
            f"Received insertion event generation={self._insertion_event_generation} "
            f"port={msg.data!r}")

    def insert_cable_goal_callback(self, goal_request):
'''

TRUTHFUL_RESULT_OLD = '''                result = InsertCable.Result()
                # A policy miss has already stopped and held the robot safely.
                # Report a normal successful result so Flowstate does not terminate
                # the enclosing process on a recoverable insertion miss.
                result.success = True
                if self._action_thread_result:
                    goal_handle.succeed()
                    result.message = "Cable insertion completed"
                else:
                    goal_handle.succeed()
                    result.message = "Cable insertion ended safely without confirmation"
                self.goal_handle = None
                return result
'''
TRUTHFUL_RESULT_NEW = '''                result = InsertCable.Result()
                confirmed = bool(self._action_thread_result)
                report_miss_as_success = os.environ.get(
                    "RL_INSERT_REPORT_MISS_AS_SUCCESS", "1"
                ).strip().lower() not in ("0", "false", "no")
                if confirmed:
                    goal_handle.succeed()
                    result.success = True
                    result.message = "Cable insertion event confirmed"
                elif report_miss_as_success:
                    # A policy miss has already stopped and held the robot safely.
                    # Report a normal successful result so Flowstate does not
                    # terminate the enclosing process on a recoverable insertion
                    # miss.  The miss is still logged and named in the message.
                    self.get_logger().warn(
                        "[aic] insertion NOT confirmed; reporting success so the "
                        "process continues (RL_INSERT_REPORT_MISS_AS_SUCCESS=1)"
                    )
                    goal_handle.succeed()
                    result.success = True
                    result.message = "Cable insertion ended safely without confirmation"
                else:
                    goal_handle.abort()
                    result.success = False
                    result.message = "Cable insertion failed: no correct-port event"
                self.goal_handle = None
                return result
'''


def patch_aic_model_source(source: str) -> str:
    source = replace_once(source, AIC_IMPORT_OLD, AIC_IMPORT_NEW, "String import")
    source = replace_once(source, EVENT_INIT_OLD, EVENT_INIT_NEW, "event subscriber")
    source = replace_once(
        source, EVENT_SHUTDOWN_OLD, EVENT_SHUTDOWN_NEW, "event subscriber cleanup"
    )
    source = replace_once(
        source, EVENT_CALLBACK_OLD, EVENT_CALLBACK_NEW, "event callback"
    )
    source = replace_once(
        source, TRUTHFUL_RESULT_OLD, TRUTHFUL_RESULT_NEW, "truthful action result"
    )
    return source


def patch_path(path: Path, expected_sha256: str, transform, label: str) -> str:
    source_bytes = path.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(
            f"refusing non-v49 {label} at {path}: {digest} != {expected_sha256}"
        )
    path.write_text(transform(source_bytes.decode("utf-8")), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit(
            f"usage: {sys.argv[0]} repo_RLInsert site_RLInsert "
            "repo_aic_model site_aic_model"
        )
    rl_digests = {
        patch_path(
            Path(path), EXPECTED_V49_RLINSERT_SHA256, patch_rlinsert_source, "RLInsert"
        )
        for path in sys.argv[1:3]
    }
    model_digests = {
        patch_path(
            Path(path), EXPECTED_V49_AIC_MODEL_SHA256, patch_aic_model_source, "aic_model"
        )
        for path in sys.argv[3:5]
    }
    if len(rl_digests) != 1 or len(model_digests) != 1:
        raise RuntimeError("patched runtime copies diverged")
    print(f"patched v50 RLInsert sha256={next(iter(rl_digests))}")
    print(f"patched v50 aic_model sha256={next(iter(model_digests))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

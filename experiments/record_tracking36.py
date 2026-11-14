"""Write a measured checkpoint from the completed V31 archive."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
batch=json.loads((ROOT/'.artifacts/repair36/tracking36v1_completed.json').read_text())
result=batch['results'][0]
assert result['verified_success'] and result['high_rate_wrist_audit']['complete_action_coverage']
note="""Latest completed result, November 3: V31 `tracking36v1` SFP **1/1** on
the selected previously failing development case `dev_45002`. Independent
terminal depth **45.845407-45.867269 mm**, lateral error **0.002890-0.018865 mm**,
angle **0.042741-0.043818 deg**; complete 2000 Hz raw peak **3.005526 N**, zero
samples over the original 8 N ceiling. Terminal dwell and source-clearance
seat checks passed against the qualified matching collision positive control.
This is a geometrically stable seat while held, not latch/electrical/release
retention validation. Duration **686.127787 wall /79.466 sim s** including
configuration. Terminal static-drag integral was **0.221127 N**. This supports
the diagnosed tracking repair on one known case; do not pool with V25-V30 or
describe it as final robustness. Full frozen batch:
`.artifacts/repair36/tracking36v1_completed.json`.

Actual headless camera footage was encoded, decoded without errors, and
inspected at middle/terminal frames:
`experiments/evidence/original_repair/video/sfp_tracking36v1_verified/attempt.mp4`.
Frames are sparse captures held causally between captures, without fabricated
interpolation. The gripper partly occludes the close view; independent recorded
geometry and force coverage complement the visual evidence. Truth overlays
are diagnostic-only and never policy inputs.

V32 `fast36v1` is a separate development test with the user-authorized 20 N
ceiling. It schedules 5 mm/s travel and 1 mm/s terminal motion, adds explicit
100 Ns/m damping feedforward plus bounded 0.5 N static-drag correction, and
retains 0.8/0.3 mm pose leads. Combined extra force is bounded at 1.2 N, with
the existing inner 6 N guard and 3 N insertion pause still active. This does
not command a 20 N push. The evaluator owns the prospective ceiling and checks
complete full-rate coverage; old archives default to 8 N. Fifty selected tests
passed after supplying both local package paths; the first invocation had
three import-path errors, resolved without changing policy code. Physical
speed benefit remains unmeasured until the new trial completes. Contact gates,
geometric tolerances and dwell requirements are unchanged.

An undeployed fixed low/high 1 N excitation candidate has synthetic controls:
0/32 no-reaction noise acceptances and 32/32 one-newton-reaction acceptances
for fixed seeds 7000-7031, plus duplicate/stale/moving/no-block rejections.
It retains the 25 mN minimum rise and adds a 3-SEM stationary contrast margin
with a source-noise floor. These are ideal sensor results, NOT Gazebo trials
or a calibrated probability. It remains outside the live frozen policy.
The old lip-gate noise limitation below therefore remains unresolved in V32.
No held-out final trial is opened; combined SC/SFP paper readiness stays false.

"""
for name in ['EXPERIMENT_PLAN.md','EXPERIMENTS.md','PAPER_RESULTS.md','RESEARCH_PROGRESS.md']:
    path=ROOT/name;first,rest=path.read_text(encoding='utf-8').split('\n',1)
    path.write_text(first+'\n\n'+note+rest.lstrip('\n'),encoding='utf-8')
p=ROOT/'paper/STATUS_FOR_AUTHORS_20261103.md'
first,rest=p.read_text(encoding='utf-8').split('\n',1)
p.write_text(first+'\n\n'+note+rest.lstrip('\n'),encoding='utf-8')

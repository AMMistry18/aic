"""Update current narrative without modifying frozen trial sources or archives."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
text="""November 3, 2026 current checkpoint (supersedes historical running statuses below):
V30 `sampling36v1` completed SFP 0/1: both lip and side-wall measurements
passed, but the 720 wall-second action deadline stopped insertion. Independent
terminal depth was 43.397542-44.077837 mm versus the 45.8 mm target; complete
2000 Hz force peak was 3.209270 N. Duration: 736.377079 wall /80.558 sim s
including configuration. Archive: `.artifacts/repair36/sampling36v1_completed.json`.
Own-observation traces show terminal drag could stall the old controller near
44 mm before its command-at-goal integral activation; terminal mode could also
chatter. V31 `tracking36v1` is testing bounded axial feedforward throughout
travel and latched terminal braking, with unchanged pose leads and 8 N audit.
Its outcome remains pending; 55 selected regressions passed before launch.

An ideal sensor-negative control exposed a separate lip-detector limitation:
with the public sensor's 0.2 N Gaussian noise and physical reaction disabled,
31/32 fixed seeds passed the old 25 mN reaction-rise gate. This is NOT a
Gazebo insertion false-positive rate; it shows that the reaction-rise check
alone does not certify contact under this noise model. Independent geometric
seat verifications remain separate. Preserve this negative result and require
controlled excitation / stationary uncertainty checks before claiming reliable
force-based lip identification. The simple unload/reload candidate remains
undeployed. Artifact: `.artifacts/repair36/lip_noise_negative_control.json`.

The user authorizes up to 20 N for future trials. Adopt a new explicitly
labelled prospective force protocol; do not change the running 8 N freeze or
relabel prior results. The ceiling is a tared resultant wrist force limit,
not a nominal push target. Preserve full 2000 Hz coverage and unchanged depth,
lateral, angular, penetration and dwell checks. Stronger useful bounded control
and faster measurement/travel must be tested, rather than merely raising aborts.
No held-out final trial has run. Combined SC/SFP robustness is not qualified.

"""
for name in ['EXPERIMENT_PLAN.md','EXPERIMENTS.md','PAPER_RESULTS.md','RESEARCH_PROGRESS.md']:
    path=ROOT/name
    original=path.read_text(encoding='utf-8')
    first,rest=original.split('\n',1)
    path.write_text(first+'\n\n'+text+rest.lstrip('\n'),encoding='utf-8')

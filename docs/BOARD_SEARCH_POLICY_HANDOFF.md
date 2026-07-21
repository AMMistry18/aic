# Board-search policy handoff

Date: 2026-07-20

Branch: `board-search`

## Current deployment: r39 one-crossing angular relief and SFP guard (2026-07-21)

Flowstate currently reports:

```text
organization: tar-2@xfa-prod-aic-us
solution:     9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH
cluster:      vmp-efe2-cf8sn65n
state:        SOLUTION_STATE_RUNNING_IN_SIM
asset:        ai.tar2.check_board_visibility_skill_v4.0.0.1+0daadf2edf3d773a20714a45a7119702cca3b23243bad23e7de6ab96c10824b3
image tag:    flowstate:check-board-visibility-v4-r39-one-crossing-sfp-guard
image ID:     sha256:a088163701b2ad74449adff4eee0c8b3cb7d323cf7cf75e7ba1a029537edc966
image tar:    e8e9675ddcdf46d47b691aa2f9d6fda2b26c8f392f48a16d63fbb0255575ae72
descriptor:   f030faf3d3c09a825c0520988416cfd59bb37f1debcea50354bd280c18603ccf
bundle:       26df83063a2d63a452a1999ed9c9a7547e440b96e032f356533461f464300f65
```

r39 fixes two failures demonstrated by the right-camera image from the r38
run. First, the embedded mask traced the black camera housing but did not fully
protect the rendered gray adapter/finger at the close oblique pose. SFP
completion and steering now use an additional `0.08 * min(image dimension)`
guard around the calibrated mask (about 82 px on a 1024-high stream) without
changing board segmentation. The pictured bottom-obscured SFP band therefore
reports gripper overlap and cannot complete.

Second, any measured J1 crossing of the target now immediately routes through
a J2-J4 clearance/posture action; the crossing counter can no longer age out
and permit repeated left/right commands. J6 similarly permits one measured
crossing, then issues J2-J4 clearance rather than reversing across the aligned
axis. This enforces at most one angular crossing before going up.

Verification: `184 passed`; the final Linux image byte-compiled the policy and
reported `sfp_guard 0.08`; `inctl skill list` returned the exact r39 asset; and
the solution remained `RUNNING_IN_SIM` on `vmp-efe2-cf8sn65n`.

## Superseded r38 synchronized SFP equipment-band recovery (2026-07-21)

r38 previously reported:

```text
organization: tar-2@xfa-prod-aic-us
solution:     9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH
cluster:      vmp-efe2-cf8sn65n
state:        SOLUTION_STATE_RUNNING_IN_SIM
asset:        ai.tar2.check_board_visibility_skill_v4.0.0.1+1d01be92dae53b1f979c970749eb9f7da7966dd0848033b3ab5e6440fa6bce28
image tag:    flowstate:check-board-visibility-v4-r38-sfp-shared-clearance
image ID:     sha256:709cc42975e304cf2ad84a3a6339bb25215cffa1f484611d4874d61743c0ff83
image tar:    970f16a29eb24d9d8ec77a2eeae031967e392342a339181ac211634fc5dd2e53
descriptor:   f030faf3d3c09a825c0520988416cfd59bb37f1debcea50354bd280c18603ccf
bundle:       d91f7920a04a87eb8704d2b3727b3f6f1893501b550be03049cab830a8392627
```

r38 restores the useful parts of the historical full-board controller for the
staged SFP target without requiring unrelated empty plate. The SFP equipment
band must now have physical image margin on every edge in all three cameras;
a single cropped band edge can be the missing fifth module and therefore
cannot complete the policy. Its r34-proven scale/coverage envelope is restored:
maximum area `0.35`, center coverage `0.96`, side coverage `0.92`, and at least
`8 px` calibrated gripper clearance.

The SFP and NIC targets now share one coherent occlusion controller. A blocked
side view moves the component row upward through the center-camera J2-J4 axes,
validates aggregate overlap/clearance on a fresh three-camera frame, and uses
shared optical backoff when that motion does not improve visibility. It never
uses opposing left/right camera axes for these rows. Missing or cropped SFP/NIC
equipment in a side camera continues shared recovery and cannot become DONE or
terminate merely because a visual-correction counter was spent; only the
synchronized predicate, the outer deadline, or a physical safety/workspace
guard can end that path.

Verification: `181 passed`; byte compilation occurred in the final Linux
image; container smoke import reported the exact SFP profile above; `inctl
skill list` returned the exact r38 asset; and the solution remained
`RUNNING_IN_SIM` on `vmp-efe2-cf8sn65n`.

## Superseded r37 NIC shared-occlusion relief (2026-07-20)

r37 previously reported:

```text
organization: tar-2@xfa-prod-aic-us
solution:     9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH
cluster:      vmp-efe2-cf8sn65n
state:        SOLUTION_STATE_RUNNING_IN_SIM
asset:        ai.tar2.check_board_visibility_skill_v4.0.0.1+801a73346f77fe25e7f740aa14eb59b6f423d657954b54b125ed3989d05edd81
image tag:    flowstate:check-board-visibility-v4-r37-nic-shared-clearance
image ID:     sha256:bd3982bc6ba1134d9442c4d38419f6a647ab787690c3f2ec868a233889e18767
image tar:    d6a023438d6826a6db7f771af814482754c39131406fb89549c1b7255bb1429b
descriptor:   f030faf3d3c09a825c0520988416cfd59bb37f1debcea50354bd280c18603ccf
bundle:       dfa67835cd2e9112c21942e3fc914bb42c4c713febdb162aa02aeb3067dfbdce
```

r37 retains r36's bounded J1/J6 yaw controller and changes only the NIC
occlusion response. The failed trace contained four clean consecutive rails,
so the lattice filter was working; the missing fifth rail was hidden from a
side camera by the gripper. Previously, whichever side camera looked worst
could independently request a camera-plane correction. Because left and right
camera axes oppose each other, those requests could reverse J2-J4 and create
the observed front/back roll oscillation.

NIC targeting now uses one shared center-camera response for all three views:

1. If the gripper overlaps the NIC component envelope in the center or either
   side camera, J2-J4 move the NIC row upward in all three images while holding
   the established J1/J6 orientation.
2. The next fresh frame must reduce total three-camera gripper overlap or
   improve the worst clearance. A successful response continues in the same
   direction; it never reverses roll because another side camera disagrees.
3. If that response does not improve visibility, the next bounded action adds
   shared optical standoff instead of reversing J2-J4.
4. Corrections are capped at four. NIC success is also stricter than r36:
   center coverage `0.88`, side coverage `0.84`, and at least `8 px` gripper
   clearance. SFP target parameters and motion behavior are unchanged.

Verification: `179 passed`; Python byte compilation and `git diff --check`
passed; container smoke import reported the intended NIC profile and reached
port 8003; `inctl skill list` returned the exact r37 asset above; and the
solution remained `SOLUTION_STATE_RUNNING_IN_SIM` on `vmp-efe2-cf8sn65n`.

## Superseded r36 bounded yaw-relief controller (2026-07-20)

r36 previously reported:

```text
organization: tar-2@xfa-prod-aic-us
solution:     9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH
cluster:      vmp-efe2-cf8sn65n
state:        SOLUTION_STATE_RUNNING_IN_SIM
asset:        ai.tar2.check_board_visibility_skill_v4.0.0.1+0a2f77f3e8cd7acf09dd6f694ae6a77364941517ba7c42b5c3edc3cee34179c5
image tag:    flowstate:check-board-visibility-v4-r36-yaw-relief
image ID:     sha256:607f251fcfb3363233062a056365c60227120f66515c0105357543b3e45ee9e0
image tar:    8f2e166ae288b1551d87ced24ab7e3199524125c69e793768f966bee21c28ecb
descriptor:   f030faf3d3c09a825c0520988416cfd59bb37f1debcea50354bd280c18603ccf
bundle:       042b7b7f1241565d1141ea5fa59db2a2b8ecfa0584aa20df4ecd143f04e67491
```

r36 retains r35's J1 -> J6 -> J2-J4 order but prevents the tighter J1 gate
from becoming a limit cycle. SFP/NIC start with a `0.18` horizontal deadband.
If a measured J1 correction crosses to the other side while both residuals
remain outside that deadband, the planner issues one `0.75`-scale base-+Z
J2-J4 clearance/roll posture step and then explicitly resumes CENTER from a
fresh image. At most two such relief moves are allowed. Their measured
overshoots add per-run hysteresis `0.18 -> 0.22 -> 0.26`; this remains far
tighter than the old `0.75` target gate but terminates the oscillation.

Coarse J1 corrections remain larger, while corrections within twice the
active deadband may use a smaller `0.10` command scale to avoid overshoot.
Yaw-response evidence is cleared whenever J6 or Cartesian motion intervenes,
so the detector cannot falsely attribute another joint's motion to J1.

Verification: `178 passed`; Python byte compilation and `git diff --check`
passed; the final container reported the intended initial `0.18` gate and
reached port 8003; `inctl skill list` returned the exact r36 asset above; and
the solution remained `SOLUTION_STATE_RUNNING_IN_SIM`.

## Superseded r35 yaw-first SFP/NIC survey (2026-07-20)

The superseded r35 deployment reported:

```text
organization: tar-2@xfa-prod-aic-us
solution:     9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH
cluster:      vmp-efe2-cf8sn65n
state:        SOLUTION_STATE_RUNNING_IN_SIM
asset:        ai.tar2.check_board_visibility_skill_v4.0.0.1+e507a5842cfbf211f026a8a4e04af4404763fff55d6249597bfc237d2e887092
image tag:    flowstate:check-board-visibility-v4-r35-yaw-first
image ID:     sha256:8cf59d24b772e75d43beb435d1510a0936324f74183b8686acfc2365797ceba8
image tar:    38a92146fa60affcde57db31fb40dcba97e05a8f015b5fc80edca576d2ca3f1f
descriptor:   f030faf3d3c09a825c0520988416cfd59bb37f1debcea50354bd280c18603ccf
bundle:       8c8f4118f04798094ea0152005e9e8bb8786f3b72fd7f04ed7ef6a0f37c28aa3
```

r35 keeps the movable-rail component envelope and fixes the motion hierarchy:

1. J1 performs coarse target centering with a `0.12` normalized horizontal
   gate (formerly `0.75`) and larger bounded proportional steps.
2. J6 confirms and trims the board long axis to `2 deg` for both SFP and NIC
   (formerly target mode could widen this to `20 deg`).
3. Only then do J2-J4 apply the required target tilt: `10 +/- 4 deg` for SFP
   and a reduced `18 +/- 4 deg` for NIC.

Post-level horizontal drift now returns to J1 then J6 rather than using a
J2-J4 camera-plane translation. An ambiguous long-axis estimate first changes
azimuth with a bounded J1 probe rather than raising the arm. After yaw is
correct, an oversized/cropped SFP or NIC view uses bounded optical backoff
instead of another +Z clearance move. The r34 full-board 8 px border gate was
removed because it drove unnecessary height; SFP completion again uses the
sliding-module equipment envelope and synchronized three-camera coverage.

Verification: `176 passed`; Python byte compilation and `git diff --check`
passed; the final container reported the intended J1/J6 limits and reached
port 8003; `inctl skill list` returned the exact r35 asset above; and the
solution remained `SOLUTION_STATE_RUNNING_IN_SIM`.

## Superseded r34 sliding-rail SFP envelope (2026-07-20)

The superseded r34 deployment reported:

```text
organization: tar-2@xfa-prod-aic-us
solution:     9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH
cluster:      vmp-efe2-cf8sn65n
state:        SOLUTION_STATE_RUNNING_IN_SIM
asset:        ai.tar2.check_board_visibility_skill_v4.0.0.1+08e6ec6a97a8720a311fadf1c88b23232bcf5c889bea954a7e6ec671a3d30126
image tag:    flowstate:check-board-visibility-v4-r34-sfp-rail-envelope
image ID:     sha256:ea4fe7c3aa226ab2c47b4e2a3cca0df4704b327418b94a3af04d2ec694293fc7
image tar:    051af6b460a437f473930fe64448d13a1c61e8c1d1a139ea8507bf6de511aa89
descriptor:   f030faf3d3c09a825c0520988416cfd59bb37f1debcea50354bd280c18603ccf
bundle:       509d57f5a029695506bd5b72795554668d4d64dc0b5617f8be6a024ef96ab2a9
```

r34 starts from the verified r32 image and changes only the SFP survey
geometry. It deliberately does **not** encode five slot coordinates because
the modules may slide. Instead, every synchronized center/left/right frame
must contain:

- the complete task-board silhouette with at least an 8 px physical border;
- the complete full-width equipment band beyond the board edge opposite the
  purple insignia, with no target-context edge contact; and
- no calibrated gripper-mask overlap with that equipment band.

The SFP center view is now `10 +/- 4 deg` from frontal, its projected target
area is constrained to `0.065 .. 0.350`, and required target coverage is
`0.96` in the center and `0.92` in both side cameras. The lower maximum scale
backs away from the too-close view that made side-camera parallax stack the
five modules. NIC and SC retain their independent r32 profiles unchanged.

Verification: `174 passed`; Python byte compilation and `git diff --check`
passed; the final container imported the deployed profile and its service
smoke test reached port 8003; `inctl skill list` returned the exact r34 asset
above; and the solution remained `SOLUTION_STATE_RUNNING_IN_SIM`.

## Superseded rollback deployment: r32 (2026-07-20)

At the user's request, the existing v4 skill was rolled back from r33 to the
last packaged revision with three independent survey geometries. Flowstate
currently reports:

```text
organization: tar-2@xfa-prod-aic-us
solution:     9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH
cluster:      vmp-efe2-cf8sn65n
state:        SOLUTION_STATE_RUNNING_IN_SIM
asset:        ai.tar2.check_board_visibility_skill_v4.0.0.1+41fa06ecef154ca8ec0765a14187d5ed0d7efdbd4cd825d4e2dc8bf3a6d41f93
bundle:       76a27d13bd58c718a991a67a773f435a65ca3717b8509c8015345b657f4e79fb
```

The packaged r32 image was inspected immediately before rollback and contains
these distinct profiles:

| Target | Center tilt | Target-area band | Center/side coverage |
| --- | ---: | ---: | ---: |
| `STAGED_SFP_MODULE` | `16 +/- 4 deg` | `0.065 .. 0.450` | `0.82 / 0.78` |
| `NIC_SFP_DESTINATION` | `28 +/- 3 deg` | `0.060 .. 0.420` | `0.82 / 0.78` |
| `SC_DESTINATION_PORT` | `26 +/- 3 deg` | `0.022 .. 0.220` | `0.82 / 0.78` |

The later r33 source experiment remains in the uncommitted working tree for
comparison, but **is not deployed**. Any next policy package should start from
the verified r32 image/behavior and optimize the SFP five-mount view without
reintroducing r33's shared-geometry changes.

## Superseded r33 canonical-IVM-view revision (2026-07-20)

The current uncommitted working-tree revision is installed as the existing v4
skill in the running **Work on this** solution:

```text
organization: tar-2@xfa-prod-aic-us
solution:     9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH
cluster:      vmp-efe2-cf8sn65n
asset:        ai.tar2.check_board_visibility_skill_v4.0.0.1+1c34e695c31544738da08ce79df2894d986df4ca454eeac5db2b0991e1726019
image tag:    flowstate:check-board-visibility-v4-r33-canonical-ivm-view
image ID:     sha256:18baef1e5aa196ecaf69c4c3310f679edcd6b6c23424a70dd888a5ceeefca386
image config: sha256:29a9fde5a0e2c7e304906cb07e40e35d2175de5276d34a33ae62a06c5eb9ca52
image tar:    395153956b6773788284555d56b27c5281a063d558fc5307d83cc048a3b141f3
descriptor:   f030faf3d3c09a825c0520988416cfd59bb37f1debcea50354bd280c18603ccf
bundle:       822c9b815230b2f08f2ea4e41374aa6dbb183f92ae99eb78598a2f4d8d1a1901
```

r33 makes the organizer's successful example view the physical goal for all
three explicit modes. The enum modes remain separate for ROI diagnostics and
future tuning, but they now share one camera geometry:

| Requirement | Shared value |
| --- | ---: |
| Center physical tilt | `16 +/- 3 deg` |
| Center board area | `0.30 .. 0.46` |
| Center target coverage | `>= 0.90` |
| Side target coverage | `>= 0.65` |
| Center target offset | `<= 0.48` normalized per axis |
| Stable synchronized frames | `2` |

The center board projection, rather than the differently sized component ROI,
is now the standoff authority. This specifically prevents an SC invocation
from accepting the preceding distant NIC pose without moving: the live stale
pose had board area `0.252` at `28 deg`, so it now requests an approach toward
the shared `16 deg`, `0.30 .. 0.46` window. It also preserves the closer NIC
view: the r32 trace reached board area `0.446`, then backed away to `0.252`
solely because an offset camera touched the top/right edge.

For explicit modes, target-mask overlap is now measured on the actual
projected component ROI. The padded ROI remains available only for clearance
and escape direction; it can no longer create tens of thousands of false
contact pixels and drive the camera away from already-visible hardware.
Missing or edge-cropped side-camera component evidence is repaired with a
small lateral camera-plane translation rather than an automatic optical
backoff.

Verification for r33: `174 passed` locally and inside the final Linux image;
Python byte compilation and `git diff --check` passed; the service smoke test
reached port 8003; `inctl skill list` returned the exact r33 asset above; and
the solution remained `SOLUTION_STATE_RUNNING_IN_SIM` on
`vmp-efe2-cf8sn65n`.

The prior r32 asset remains available for rollback:

```text
ai.tar2.check_board_visibility_skill_v4.0.0.1+41fa06ecef154ca8ec0765a14187d5ed0d7efdbd4cd825d4e2dc8bf3a6d41f93
```

## Superseded r32 component-authority revision (2026-07-20)

The current uncommitted working-tree revision is installed as the existing v4
skill in the running **Work on this** solution:

```text
organization: tar-2@xfa-prod-aic-us
solution:     9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH
cluster:      vmp-efe2-cf8sn65n
asset:        ai.tar2.check_board_visibility_skill_v4.0.0.1+41fa06ecef154ca8ec0765a14187d5ed0d7efdbd4cd825d4e2dc8bf3a6d41f93
image tag:    flowstate:check-board-visibility-v4-r32-component-authority
image ID:     sha256:f3a219c70f19cd15081c455f7aba75f248a2bdae68ac1745cbe60ecd5b9f0b2f
image config: sha256:4cc8316e1267c15dc44805284b91a94f4519ce18efb9588cd56d176a76c4d62c
image tar:    199953e65c7a7578c13284cf04c429223464e934bb71e07b999b8307bc25199b
descriptor:   f030faf3d3c09a825c0520988416cfd59bb37f1debcea50354bd280c18603ccf
bundle:       76a27d13bd58c718a991a67a773f435a65ca3717b8509c8015345b657f4e79fb
```

r32 fixes the extra motion seen in the r31 traces by making the selected
component ROI authoritative in explicit target modes. A fully visible target
is no longer rejected merely because the whole-board context pad touches one
image edge, the plate's long-axis estimate is temporarily ambiguous, or the
board is not centered as tightly as the component itself. Opposite-edge crop,
target scale, target coverage, requested physical tilt, and gripper overlap
remain hard failures.

The explicit-mode completion profiles are now:

| Target | Center tilt | Center ROI coverage | Side ROI coverage | Target-center limit | Roll limit |
| --- | ---: | ---: | ---: | ---: | ---: |
| `STAGED_SFP_MODULE` | `16 +/- 4 deg` | `>= 0.82` | `>= 0.78` | `0.75` normalized | `20 deg` |
| `NIC_SFP_DESTINATION` | `28 +/- 3 deg` | `>= 0.82` | `>= 0.78` | `0.75` normalized | `20 deg` |
| `SC_DESTINATION_PORT` | `26 +/- 3 deg` | `>= 0.82` | `>= 0.78` | `0.75` normalized | `20 deg` |

The planner now steers from the requested component center, keeps a close
target view when the whole-plate axis is ambiguous, and suppresses zoom-out or
clearance moves caused only by ignored whole-board padding. Each camera log
also prints the target identity, actual/padded ROI coverage, scale, center,
edge contacts, mask overlap, and mask clearance so future failures no longer
end with an unexplained `center-camera survey predicate still fails` message.

Verification for r32: `172 passed` inside the final Linux image; Python byte
compilation and `git diff --check` passed; the service smoke test reached port
8003; `inctl skill list` returned the exact r32 asset above; and the solution
remained `SOLUTION_STATE_RUNNING_IN_SIM` on `vmp-efe2-cf8sn65n`.

The prior r31 asset remains available for rollback:

```text
ai.tar2.check_board_visibility_skill_v4.0.0.1+a408653caeed0f1d845afc06e8990e232cac07a2d6974cb5de5cf3110ebb145d
```

## Superseded r31 target-geometry revision (2026-07-20)

The uncommitted working-tree revision is installed as the existing v4 skill in
the running **Work on this** solution:

```text
organization: tar-2@xfa-prod-aic-us
solution:     9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH
cluster:      vmp-efe2-cf8sn65n
asset:        ai.tar2.check_board_visibility_skill_v4.0.0.1+a408653caeed0f1d845afc06e8990e232cac07a2d6974cb5de5cf3110ebb145d
image tag:    flowstate:check-board-visibility-v4-r31-target-geometry
image ID:     sha256:c922125d4f7447c002ec263e23150b820d40f6e41b8a3c3b6d421c5e455263ab
image tar:    393e1542dfa9c9b7c7779e17900b507670bda7ebf3d4c0e060f553a6cfb4c6ef
descriptor:   f030faf3d3c09a825c0520988416cfd59bb37f1debcea50354bd280c18603ccf
config:       11670db3477433adc20c59f2ab826cb7132359596d0749a0376dcf41bf35e74b
bundle:       51e6a1ac754b59ece0f0e4a071094f1f8f15a586701b924d876ab4c0207f635f
```

Flowstate exposes `survey_target` as a dropdown with three explicit modes:

| Mode | Required IVM region |
| --- | --- |
| `STAGED_SFP_MODULE` | Staged SFP cable/module source |
| `NIC_SFP_DESTINATION` | Five NIC cards and SFP destination ports |
| `SC_DESTINATION_PORT` | Five blue SC destination ports |

`UNSPECIFIED` remains enum value zero and preserves the legacy board-overview
behavior for existing saved nodes. Explicit modes do not require the entire
plate to satisfy the old padded full-board goal. They derive a board-relative
component ROI and steer, mask, and complete against that ROI. The selected ROI
must pass context and calibrated gripper-clearance predicates in the center,
left, and right cameras for two synchronized snapshots; no single camera can
complete the policy.

The intended Flowstate sequence is:

```text
v4(STAGED_SFP_MODULE)
  -> Switch To Default Controller
  -> save current root_t_tool0 as saved_sfp_survey_pose
  -> SFP-module IVM/filter/belief
v4(NIC_SFP_DESTINATION)
  -> Switch To Default Controller
  -> NIC/SFP destination IVM/filter/belief
  -> complete the entire SFP insertion
Move Robot back to saved_sfp_survey_pose
  -> SC-plug IVM/filter/belief (no additional board search)
v4(SC_DESTINATION_PORT)
  -> Switch To Default Controller
  -> SC-port IVM/filter/belief
  -> complete the SC insertion
```

The SFP-source TCP pose is deliberately reused after SFP insertion because the
SC plug becomes visible when that staged SFP cable is removed. Run **Switch To
Default Controller after every v4 invocation** before another arm skill; the
v4 wrapper releases its invocation lock, while the switch restores the
controller expected by Move Robot. The complete node and pose-output wiring is
in [V4_TARGET_SURVEY_FLOWSTATE_WIRING.md](V4_TARGET_SURVEY_FLOWSTATE_WIRING.md).

Downstream execution must still require:

```text
success == true AND done == true AND component_coverage_ready == true
```

Verification for r31: `170 passed`; Python byte compilation and package checks
passed; the service smoke test reached port 8003; `inctl skill list` returned
the exact asset above; and the solution remained running in simulation on
`vmp-efe2-cf8sn65n`.

## Deployed target-view tuning

The r30 logs showed that the component ROI could be visibly usable while v4
continued moving because `target_region_full` treated a **steering-only
20--48 px context pad** as a hard completion condition. The next local
revision keeps synchronized center/left/right component coverage and zero
gripper overlap, but evaluates the projected hardware coverage itself rather
than requiring that extra empty border.

| `survey_target` | Center physical tilt | Why |
| --- | ---: | --- |
| `STAGED_SFP_MODULE` | `16 +/- 4 deg` | Nearly frontal view: exposes the SFP module top/retention geometry in all three cameras. |
| `NIC_SFP_DESTINATION` | `28 +/- 3 deg` | Moderate oblique view: preserves card top and side/depth cues. |
| `SC_DESTINATION_PORT` | `26 +/- 3 deg` | Moderate oblique view: preserves connector opening and local depth cues. |

For each explicit target mode, the center camera must retain at least 95% of
the projected target ROI, the side cameras at least 92%, and all three must
remain unmasked and inside the target's IVM scale band for two synchronized
frames. A single context-pad edge is now acceptable when the actual ROI is
still sufficiently visible; opposite-edge cropping remains a rejection. This
is intentionally not a relaxation to a one-camera or partial-target success
condition.

The target-specific profile overrides the legacy whole-board tilt fields for
explicit modes. `UNSPECIFIED` continues using the configured proto/default
whole-board geometry. This change is packaged and installed as the r31 asset
above. The prior r30 asset remains available for rollback:

```text
ai.tar2.check_board_visibility_skill_v4.0.0.1+eee535a4aa46b869309ffd8c4a657952ca1222642c0efc9c847e171d537c49cc
```

## Superseded r29 and r28 revisions

r29 and r28 are historical and must not be treated as the installed policy.
r29 was the immediately preceding whole-component-survey iteration:

```text
ai.tar2.check_board_visibility_skill_v4.0.0.1+5a54c65fbc0a6e45e65aeab2c7573a493ee4b415887d80e01a57c694232f7919
```

Its trace oscillated between vertical framing and center gripper-mask escape,
then reached the 90-second deadline without a terminal synchronized survey.
r30 replaces that single global goal with the three target-specific ROIs above.

The r28 revision below is also superseded. Its debugging history remains useful
for the 32-degree survey-vector, J6, clearance-order, and IK-redistribution
changes that r30 retains.

## r28 component-view design history (superseded)

This revision was previously installed in the running **Work on this** solution
but is no longer the current asset.

```text
organization: tar-2@xfa-prod-aic-us
solution:     9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH
cluster:      vmp-efe2-pq1kedw2
asset:        ai.tar2.check_board_visibility_skill_v4.0.0.1+4eee6e81a723df4cce977826ce644048fd06d4c22f7472ad856585158ef582e8
image tag:    flowstate:check-board-visibility-v4-r28-three-camera-components
image ID:     sha256:4139093059f05bfd9780b3017410794df967ddccbfd9431d7081eec5c63acfe8
image tar:    f6dc39973ca08af2ae9c05a71c7ff1b5a521892676d6dafc63a182bad059fc9b
descriptor:   e4ed55f4a3151497c2195603a198d605f40580c8a6ab53afbb0f2ff03cc326f1
config:       c8643525176c6b7ca54541dc6cd22b3aaa390ed310845a4bdeb31d5a924f2f80
bundle:       3ee60273b0ae906753560d416a4046ad4aaf029ae4988509002d9ff0465568fb
```

r28 moved away from a perfectly vertical, padded full-plate view. After J1
centers the equipment region and J6 aligns the board long axis within
2 degrees, joints 2--4 converge the center optical axis to a deterministic
oblique survey vector:

```text
target tilt from vertical:       32 degrees
accepted vector error:           +/- 2 degrees
tilt axis:                       horizontal projection of image-right
first pitch clearance:           one 0.015 m base-Z escape maximum
later pitch clearance:           0.000 m (no cumulative lift)
center projected board area:     0.32 .. 0.50
minimum side-camera area:        0.12 each
confirmation:                    2 synchronized three-camera frames
plate-boundary context margin:   at least 64 px in all three cameras
side-camera center error:        at most 0.25 on each image axis
gripper-mask separation:         at least 32 px in all three cameras
```

The vector gate constrains both magnitude and azimuth. A camera already tilted
32 degrees in the wrong direction is corrected instead of accepted. Tilting
around image-right (after J6) introduces perspective along the board short
dimension consistently, which preserves depth cues on the SC/SFP hardware and
five NIC cards without sacrificing their use of the wide image dimension.

The 0.32 lower scale gate is the board-pixel proxy for IVM's documented
0.25--0.5 m working range; the policy is not permitted to query a board/world
pose and therefore cannot measure camera-to-board distance directly. A smaller
projection commands up to three bounded optical approaches rather than more
upward clearance. The first pitch correction alone may use 15 mm of base-Z to
escape the initial singular posture. Later pitch corrections retain the
measured position apart from visual recentering, preventing the repeated 20 mm
lifts that left r25 too high. The old scale-dependent empty-border requirement
is excluded from completion. It is replaced by a fixed 64-pixel minimum around
the detected plate boundary in all three cameras. This prevents the
cable-module row from sitting on a crop edge while avoiding the excessive
zoom-out caused by the old dynamic padding. The detected plate footprint,
logo, component scale, long-axis geometry, gripper separation, and both
side-camera views remain required.

The r26 failure trace had a credible board with only its top edge clipped. The
pre-level `_yaw_cannot_help` gate incorrectly treated that as an unusable
centroid and exhausted the bounded zoom-out budget before the signed J2--J4
vertical-framing servo could run. r27 permits a credible single-top-edge or
single-bottom-edge view to finish J1/J6 using horizontal centroid and long-axis
evidence; the post-level servo then owns vertical framing and must establish
the cable-row margin before completion.

The r27 trace did not produce a terminal survey. Cartesian J2--J4 positioning
redistributed J1/J6 by roughly 0.02--0.06 rad, and the wrapper restarted
CENTER/ALIGN after every such move. That repeated J6 and leveling until a final
J6 command was force-reversed; v4 returned `success=False, done=False`. r28
keeps the measured Cartesian progress, lets fresh image evidence decide whether
centering or alignment actually changed, and permits one two-frame-confirmed
J6 trim at the final framing pose. It also corrects the center-clearance edge
mapping from the erroneous `left, top, right, bottom` order to the report's
actual `left, right, top, bottom` order, so J2--J4 corrections use the intended
image axis.

All three cameras now require the larger 64-pixel component envelope. Each side
camera must additionally stay within 0.25 normalized center error on both axes
and at least 32 pixels from its calibrated gripper mask. The worst failing side
view drives a small correction through that camera's own TF before completion.

New optional proto parameters (zero selects the defaults above):

- `target_center_tilt_deg`
- `center_tilt_tolerance_deg`
- `ivm_min_center_board_area_frac`
- `ivm_max_center_board_area_frac`

Historical r28 verification: `154 passed`, Python byte compilation passed, `git diff --check`
passed, all four new protobuf fields were present in the Linux image, the local
smoke service reached gRPC port 8003, the exact installed asset appeared in
`inctl skill list`, and the in-cluster service reached camera initialization
and gRPC port 8003.

The behavior tree must gate downstream IVM execution on all three v4 outputs:

```text
success == true AND done == true AND component_coverage_ready == true
```

The observed tree also continued into IVM after r27 returned `success=False` and
`done=False`, so the four-module filter error was reporting a failed
search view rather than a valid terminal survey. This dataflow guard is a
required workflow condition, not an additional motion safety limiter.

## Earlier revisions (all superseded)

The immediately superseded r27 component-view asset was:

```text
ai.tar2.check_board_visibility_skill_v4.0.0.1+e1ee22b3546fdb8c0a8ddf178c8543583181f1a8358ff3ff6e79fe95b7f18ee7
```

It introduced the credible single-edge transition and a 32-pixel context
margin, but restarted J1/J6 after normal Cartesian IK redistribution and used
an incorrect clearance-to-edge order for center-camera corrections.

The earlier r26 component-view asset was:

```text
ai.tar2.check_board_visibility_skill_v4.0.0.1+4bbea135e46db5579c21782329db16afa8410fed8bc29cfb3c399f07b3d5af14
```

It introduced the 32 +/- 2 degree survey vector, 0.32--0.50 center scale
window, and one-time 15 mm pitch clearance, but could terminate unsuccessfully
before the post-level vertical-framing servo when a credible view had only one
vertical edge clipped.

The earlier r25 component-view asset was:

```text
ai.tar2.check_board_visibility_skill_v4.0.0.1+e46dff7aaf6b4cb4d8ac47a9c0571a0fdefe7b0e648fb42c16edaa90ebdd0709
```

It used a 20 +/- 4 degree survey vector, a 0.28--0.44 center scale
window, and added up to 20 mm base-Z clearance on every pitch increment.

The implementation source used for the installed bundle is committed and
pushed to `origin/board-search` at:

```text
31499c7ac26dfdfd108a7c766c5dbd175911ea21
```

It was built and installed in the running **Work on this** solution on cluster
`vmp-efe2-hv8d2ahu` as the existing v4 skill:

```text
ai.tar2.check_board_visibility_skill_v4.0.0.1+872b09c0a4b9854b25b954737fce4fe8eced07bccb5d423f3497eb8ed4cb969d
```

The install completed on 2026-07-18 at approximately 23:13 CDT. The solution
remained `SOLUTION_STATE_RUNNING_IN_SIM`. No other skill or service was rebound.

```text
image tar SHA-256:  67eb2f8757e36d3406a45f93d25c503565b90c20f1570a4405607e909dc99340
bundle tar SHA-256: 543e4298f6437f8e54c0b93075ebc7d3bc0c5b9ff3ec6fea550a6db4f73cb07b
```

## Retained motion behavior in r30

The policy is a measured, camera-driven phase machine:

```text
ACQUIRE/CENTER (J1)
        -> ALIGN LONG EDGE (J6, <= 2 degrees)
        -> SELECT survey_target BOARD-RELATIVE ROI
        -> SET 32 +/- 2 DEGREE CENTER SURVEY TILT (primarily J2-J4)
        -> OPTICALLY FRAME THE SELECTED COMPONENT ROI (J2-J4)
        -> CLEAR THAT ROI IN BOTH SIDE CAMERAS (J2-J4)
        -> DONE after two synchronized target-ROI frames in all three cameras
```

Left and right cameras may help select the initial J1 direction and now provide
mandatory supporting evidence for completion. Neither side can finish the
search by itself. The center still owns J1/J6 alignment, scale, identity, and
the fresh physical survey-vector TF check; both sides must simultaneously retain
usable context and gripper separation for multi-camera IVM.

The gripper masks still exclude calibrated robot pixels before board component
selection, but they are no longer diagnostic-only. In each explicit mode, the
selected component ROI becomes the protected survey envelope. Completion
requires zero overlap between that target envelope and the mask plus the
mode-specific minimum separation. The report's measured escape vector drives
J2-J4 away from the mask. `UNSPECIFIED` retains the legacy board-overview path.

## Root cause of the no-motion run

The failing trace requested this valid correction:

```text
joint1=-0.5760
target_joint1=-0.5132
delta=+0.0628 rad
```

At the old `0.12 rad/s` profile rate, the minimum-jerk command required:

```text
1.875 * 0.0628 / 0.12 = 0.981 seconds
```

`RobotMotion` embedded the latest controller target mode inside each
asynchronously received `/joint_states` reference and expired that mode after
`0.5 seconds`. The per-sample guard then required both a fresh joint vector and
that still-fresh embedded mode. At `0.525 seconds`, it confused an aged mode
timestamp with stale measured joints, stopped the profile, and reversed the
small partial motion. This exactly explains why the UI appeared not to move.

The wrapper also selected fallbacks by searching message strings. The stale
feedback wording was not on the J1 fallback list, so the policy terminated
instead of trying its Cartesian base-yaw route.

## Historical motion fixes retained in r30

### Direct-joint transactions

File: `flowstate/aic_perception/aic_perception/robot_motion.py`

- Added machine-readable `MotionFailure` values.
- Joint feedback freshness is now checked independently from controller-mode
  freshness.
- Joint mode must still be confirmed on a newer measured sample before a
  profile starts.
- During the profile, a genuinely stale measured joint vector aborts.
- A fresh explicit controller report that mode changed away from joint mode
  aborts and reverses.
- A temporarily unknown or aged mode report does not invalidate healthy
  measured joints.
- Settling accepts newer measured joints without requiring every sample to
  carry a synchronized mode timestamp.
- Direct J1/J6 profiles accept up to `0.20 rad/s`; the controller continues to
  enforce its native URDF position and velocity limits.
- The old artificial `0.30 rad` per-command rejection was removed. The planner
  still emits incremental, remeasured commands.

### Wrapper fallbacks and timing

File: `flowstate/aic_perception/check_board_visibility_skill.py`

- Direct J1 and J6 fallbacks now use `MotionFailure`, not log-message parsing.
- Default Cartesian speed increased from `0.04` to `0.05 m/s`.
- Default Cartesian angular speed increased from `0.20` to `0.30 rad/s`.
- Default direct-joint speed can reach `0.20 rad/s`.
- Default per-command timeout changed from `8` to `6 seconds`.
- Default workflow deadline changed from `150` to `90 seconds`.
- Legacy cumulative/start-relative motion-envelope proto fields remain
  accepted for existing Flowstate node compatibility, but the viewpoint policy
  no longer uses them as termination conditions.
- Wrist force remains the hard physical runtime stop. Fresh feedback is
  required before and throughout every movement.
- Cartesian leveling/clearance that redistributes motion through J1 or J6 keeps
  the achieved framing pose. Fresh image predicates remain authoritative, and
  a residual long-axis error gets one confirmed J6 trim at the final pose.
- Leveling continues while it makes measured top-down progress; it no longer
  fails after an arbitrary five-stage count.

### J6 accuracy and strict completion

File: `flowstate/aic_perception/aic_perception/viewpoint_search.py`

- Long-axis alignment tolerance is `2.0 degrees`, inclusive.
- Long-axis ratio must be at least `1.15` before an orientation estimate is
  trusted.
- Two fresh frames must agree on the signed correction before J6 moves.
- The old minimum J6 correction was `0.15 rad` (`8.6 degrees`), which could
  never settle accurately near a two-degree target. The minimum is now
  `0.02 rad` (`1.15 degrees`).
- A correction transaction may cover up to `0.45 rad` (`25.8 degrees`) before
  the next measured image, reducing unnecessary mode switches on large errors.
- After J1, J6, and survey leveling, terminal evidence is deliberately
  rechecked in two consecutive synchronized snapshots. The center must contain
  the full board and component context, logo identity, rectangularity >= 0.72,
  32-50% board area, long-axis ratio >= 1.15, orientation error <= 2 degrees,
  32 +/- 2 degree survey TF, 64 px component context, zero protected-envelope
  mask overlap, and >= 32 px separation. Each side camera must simultaneously
  retain logo/board evidence, >= 12% area, rectangularity >= 0.55, center error
  <= 0.25 on each image axis, 64 px component context, zero mask overlap, and
  >= 32 px separation.

### Vertical J2-J4 visual servo

Files:

- `flowstate/aic_perception/check_board_visibility_skill.py`
- `flowstate/aic_perception/aic_perception/viewpoint_search.py`

The 21:52 live trace proved that geometric top-down leveling reduced camera
tilt correctly but moved the board the wrong way in the image:

```text
before leveling: center_y=+0.241, tilt=0.202 rad
after stage 1:   center_y=+0.344, tilt=0.109 rad
after stage 2:   center_y=+0.423, tilt=0.027 rad
```

The old ASCEND ordering then selected optical-axis backoff before vertical
camera-plane centering. Backoff preserved the bad lower-edge projection and
introduced enough J1/J6 IK drift to restart alignment.

The revised policy now:

- adds signed image-plane vertical correction to each top-down leveling stage;
- interprets positive center Y as a board low in frame and negative center Y
  as a board high in frame;
- prioritizes bounded bidirectional camera-plane centering through J2-J4 before
  optical backoff;
- compares the next fresh center frame with the pre-move vertical error; and
- reverses the learned image-Y polarity when the absolute error worsens toward
  the same edge, preventing repeated wrong-way J4/arm compensation.

The camera orientation target remains physically top-down during these moves;
vertical centering is not achieved by accepting a slanted terminal camera.

### Historical r28 gripper-clear survey controller

The 22:10 live run proved that the old terminal predicate was too weak. It
accepted a geometrically complete and top-down center view at area `0.291`,
rectangularity `0.77`, orientation error `0.7 degrees`, and long-axis ratio
`1.56`, but it also logged `gripper_mask_contact=True`. IVM returned estimates,
yet the NIC filter found only three physical rails because the gripper still
occluded the lower-right detail region. The rail filter correctly refused to
invent the two missing cards; score/count thresholds were not relaxed.

The revised controller adds these measured quantities to every center-camera
report and log line:

- `gripper_overlap_px`: overlap between the calibrated mask and the protected
  board/component envelope;
- `gripper_clearance_px`: minimum separation after overlap reaches zero; and
- `gripper_escape_direction`: normalized desired board-image displacement away
  from the mask.

While leveling and during the final survey phase, the camera moves opposite
that image escape vector through J2-J4. The next fresh frame validates overlap
and clearance; if separation worsens, the image-Y polarity reverses. Once the
mask is clear, the old generic vertical-centering servo no longer pulls the
board back behind the gripper unless a physical/context edge is actually
clipped or the vertical displacement exceeds 35% of the image.

The final scale controller backs away above 36% board area and makes up to
three bounded approaches below 26%. Expanded component context must clear all
four physical image edges by at least `1.25 * context_pad_px`. These constraints
keep all five NIC rails and both SC rail/port regions usable without guessing
from partial detections.

J1 centering also learns the observed horizontal image response per signed
command scale. After the first measured correction, later J1 steps use that
live response with a bounded 85% correction target, reducing repeated small
motions while retaining fresh-frame verification.

### Historical r28 synchronized full-component survey

The subsequent 22:54 IVM run captured all three cameras but returned only four
distinct NIC rail candidates (`[-142.6, -95.9, -54.3, -22.5] mm`). The center
image showed the complete board, while both oblique side images still placed
the lower NIC region against their camera-fixed gripper silhouettes. This is a
policy-exit gap, not a reason for the NIC filter to invent the missing rail.

r28 treated the three images as one synchronized terminal
contract:

- the center retains the strict 32 +/- 2 degree tilt, 2-degree long-axis,
  32-50% scale, 64 px component-context contract;
- every configured side report must independently retain logo identity, at
  least 12% board area, rectangularity >= 0.55, center error <= 0.25 per axis,
  64 px component context, zero protected-envelope overlap, and >= 32 px
  gripper separation;
- once the center is strict, the worst failing side view drives a small
  0.6-1.0 scale translation using that side camera's own image axes and TF;
- mask-escape polarity is learned independently for center, left, and right
  from the next synchronized frame; and
- any side correction that disturbs center geometry is repaired in the
  post-level framing phase; it does not restart the initial J1/J6/tilt loop.

Side reports are supporting constraints, never standalone success paths. This
preserves the center camera as the survey reference while ensuring the exact
three images submitted to IVM are simultaneously useful.

## Limits that remain

The policy intentionally retains execution correctness requirements:

- finite numeric commands;
- an available controller subscriber;
- confirmed controller mode at direct-joint transaction start;
- fresh measured feedback;
- cancellation and an overall workflow deadline;
- per-command timeouts so an unresponsive controller cannot block forever;
- native controller joint/velocity limits; and
- fresh wrist-force feedback with the configured absolute/delta force stop.

Arbitrary cumulative translation, cumulative rotation, start-relative
workspace, start-relative joint, and fixed leveling-stage limits no longer stop
the search.

## Verification

Run from the repository root:

```powershell
$env:PYTHONPATH=(Resolve-Path 'flowstate/aic_perception').Path
python -m pytest -q flowstate/aic_perception/test
```

Result for the current r30 revision:

```text
166 passed
```

The tests include regressions for:

- the exact `0.0628 rad`, `0.981 second` J1 profile continuing when the
  independent mode timestamp has aged out;
- explicit controller mode change still reversing the transaction;
- preserving all non-requested joints during direct J1 and J6 motion;
- two-degree J6 tolerance and fine correction size;
- strict phase order;
- side cameras never finishing the search by themselves;
- two consecutive synchronized three-camera frames being required;
- per-side mask/context corrections using the selected camera's image plane;
- independent left/right mask-escape polarity validation and reversal;
- target-ROI mask overlap, clearance, and escape direction;
- mask-escape polarity reversal when a fresh frame gets worse;
- bounded approach when a complete board is too small for IVM detail;
- learned J1 image response increasing centering efficiency;
- 32-degree survey-vector completion being checked from fresh TF;
- high/low vertical correction, fresh-frame polarity validation, and reversal
  after a wrong-way response; and
- controller handoff finalization order.
- stable three-mode protobuf numbering and legacy `UNSPECIFIED` behavior; and
- target-specific ROI projection, centering, visibility, and mask gates.

Also run before any future bundle:

```powershell
python -m py_compile `
  flowstate/aic_perception/aic_perception/robot_motion.py `
  flowstate/aic_perception/aic_perception/viewpoint_search.py `
  flowstate/aic_perception/check_board_visibility_skill.py
git diff --check
```

## Historical r28 expected live trace

The first retest should show:

1. J1 action planned with `control=direct_shoulder_pan_joint`.
2. Direct J1 completion with a measured joint delta close to the request.
3. A fresh center frame and further J1 correction or centering confirmation.
4. J6 sign confirmation, then direct J6 corrections.
5. J6 accepted only at `<= 2.0deg`.
6. J2-J4 top-down leveling with a logged signed vertical correction.
7. If the board is still high/low, `action=translate` occurs before backoff.
8. The next fresh frame either validates the direction or logs
   `reversing image-y polarity` and commands the opposite direction.
9. Center `gripper_overlap` reaches zero and clearance reaches at least 20 px.
10. Any failing side view logs a camera-specific `action=translate` at scale
    0.6-1.0; its next frame validates or reverses that camera's polarity.
11. Left and right simultaneously reach zero overlap, at least 12 px clearance,
    and usable component context.
12. Two consecutive synchronized survey candidates are observed, then the
    skill succeeds and releases its controller/arm resources in the existing
    finalizer.

The old message below must not recur merely at the 0.5-second mark:

```text
measured joint feedback became stale; joint 1 yaw reversed
```

If it does recur, compare actual `/joint_states` receipt timestamps rather than
loosening mode timing again; the revised message now means the measured joint
stream itself was unavailable.

## Flowstate wiring

Keep the behavior-tree controller handoff serial:

```text
Switch To AIC Controller
Check Board Visibility v4
Switch To Default Controller
Require board_result.success && board_result.done
IVM / Move Robot
```

The board-search skill publishes a measured-state hold before returning, but
`Switch To Default Controller` is still the node that releases the shared
Flowstate `arm` lease. Do not put result validation, IVM, or Move Robot before
that switch.

Before building, inspect the node's explicitly stored parameter values. A
nonzero value saved in Flowstate overrides the new source default even though
the manifest and code changed.

## Known limitations and next-agent guidance

- The r30 bundle/install and service-start smoke checks passed. The
  target-specific ROI controller still requires a task run to validate its
  physical mask-clear convergence and downstream per-target IVM yield.
- The wrapper still contains legacy envelope branches. They are inert because
  runtime envelope values are infinite after parameter validation. Removing
  that dead compatibility code is optional cleanup, not required for the next
  test.
- Do not relax the NIC filter's five-rail requirement. A missing rail in
  `NIC_SFP_DESTINATION` means that target survey still failed to expose all
  physical cards; inspect the ROI overlap/clearance/scale diagnostics first.
- Do not relax the two-degree J6 goal. If convergence oscillates, inspect the
  signed angle and measured J6 delta first.
- Do not reinstall the last asset identity expecting new code. Flowstate assets
  are content-addressed; a future authorized build must produce and install a
  new digest.

## Documentation cleanup performed

Documents tracked on `origin/main` were retained. Custom Flowstate deployment,
board-search, IVM, and perception documents were retained. Two stale custom
training handoffs unrelated to the current Flowstate policy were removed:

- `docs/ALIGN_RL_CODEX_HANDOFF.md`
- `docs/SC_PORT_TEACHER_HANDOFF.md`

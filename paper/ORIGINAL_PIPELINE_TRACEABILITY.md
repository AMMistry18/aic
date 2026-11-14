# Original article, checkpoints and Gazebo changes

Audit date: November 3, 2026. Primary article source:
[team article](https://github.com/SatyaA27/intrinsic-ai-for-industry-challenge-team-tar-2-sol/blob/main/index.md),
Git blob `6e0483ce24854aaa2aea6b28962ee0270ca31360`.
The article names the method and code; it does not pin binary checkpoint hashes.
The original tracked files were also verified against the article-linked
`AMMistry18/aic` repository at commit `fd27bbe875dbd656e381f72ae2b44efdfb243924`.
All four Git blob identities match. Our hash checks establish both the published
original identities and the subsequent adaptation/runtime lineage.

| Component | Original checkpoint | Current research checkpoint |
|---|---|---|
| SC plug | `best_sc_plug_pose.pt` | Original unchanged |
| SC mouth | `best_sc_mouth_pose.pt` | Fine-tuned from original; `sc_mouth_standard30_v1` |
| SFP plug | `best_sfp_plug_pose.pt` | Fine-tuned from original; `plug33_v1` |
| NIC/SFP ports | `best.pt` | Fine-tuned from original; `nic33_v1` |

All four original files match their tracked HEAD bytes. Training records name
the matching original SHA-256; adapted output hashes and archived runtime
overrides match. SC mouth training includes its recorded checkpoint recovery.
No held-out final cases were used for adaptation. Reproduce the checks with
`python experiments/audit_checkpoint_lineage36.py`; complete paths/hashes:
`.artifacts/repair36/checkpoint_lineage.json`.

The article's learned keypoints, confidence-weighted plug triangulation/rigid
fit, SC native crop refinement and RGB per-grasp TCP-to-tip transform remain
part of the research system. Insertion uses geometry and wrist force, not the
retired RL checkpoint. SFP fixture refinement also uses confidence-weighted
image residuals and public CAD. Fine-tuned models are not byte-identical
originals, and the new experiments must not be described that way.

The research controller is an extension with narrower scope: prescribed rigid
grasp and upstream angular alignment, RGB/CAD translation refinement, SC
shoulder or SFP entrance/side-wall datums, and bounded tracking compensation.
It does not reproduce the full survey, pick and free arm-motion sequence or
the original full-orientation SFP correction. Scoring-event success was replaced
by an independent geometry/contact/full-rate force verification contract.
Motion and force settings also differ: the article reports SFP 15/6 mm/s
free/contact speeds, whereas the qualified V33 research travel uses 5/1 mm/s
with different connector-specific registration stages. The original settings
must not be presented as the parameters used for the new measured trials.

The paper should distinguish original-checkpoint perception, adapted perception
and added control components. Existing body/tracking component comparisons do
not isolate confidence weighting; no weighting-only improvement claim is yet
supported. Keep matched contact/tracking ablations on identical adapted weights.

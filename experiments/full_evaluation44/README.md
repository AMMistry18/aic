# Combined SC/SFP full evaluation

November 13 final-study checkpoint: eight completed final policy trials
independently verified, SC 4/4 and SFP 4/4, all with complete 2000 Hz force
coverage. Maximum raw peak: SC 3.150219 N, SFP 3.404254 N. Successful action
medians: SC 28.33825 simulated /293.179765 wall seconds; SFP 58.41375
simulated /631.072275 wall seconds. These are partial final results, not a
completed reliability study.

The ninth selected case, SC 700026, never reached controller-ready or policy
execution. Its scene and logs are retained as a setup failure. A Docker-stop
30s timeout escaped cleanup and stopped the original coordinator; the initial
startup error was not persisted, so no further startup cause is asserted.
All-attempt accounting is SC 4/5 and SFP 4/4; completed-policy denominators
remain SC 4/4 and SFP 4/4. This selected case is not retried or replaced.

`experiments/resume_full_evaluation44.py` resumed at the next frozen manifest
entry, SFP 710039. It preserves the exact policy, model, physics, verifier and
manifest hashes and separately records its own coordinator source/hash. It
reconciles uncommitted outcomes and transport cleanup without changing policy
action budgets. Two bookkeeping regressions reject skipped failures, altered
grasps and duplicate execution. No preliminary simulation was added. Current
status and partial results remain in `experiments/full_evaluation44/progress.json`;
91 primary cases including the active trial and 80 ablations remain. The
manuscript's older snapshot is preserved until a reviewed results update.

November 12 final-study launch: at the user's explicit request, remaining
preliminary simulations were cancelled and the recovered baseline entered the
proper final evaluation. `--start-final-now` froze source/checkpoint hashes and
a 100-case manifest before any final outcome: 50 SC and 50 SFP, followed by
80 paired ablation trials on a 40-case subset fixed before outcomes. The first
selected case is SC 700029. Four development ablation simulations were skipped.
The contact-control simulation had already completed; its retained recording
qualified with 1334 physical requested contact pairs, without another run.
All per-trial setup, independent seating/contact checks, continuous dwell and
complete 2000 Hz force auditing with the 20 N measured ceiling remain enabled.
No final-source changes, tuning, replacement of failures or success-dependent
stopping are permitted. Freeze/manifest and live results are preserved under
`experiments/full_evaluation44/`. Earlier preliminary-check requirements below
are superseded by this recorded user instruction. The manuscript's older
snapshot is unchanged; final outcomes will be reported separately.

November 12 update: the user stopped speed43 after five completed development
trials (SC 0/3, SFP 1/2 verified). The sixth trial was cancelled. The fifth
trial's complete 2000 Hz force audit measured 109.895620 N, exceeding 20 N;
its failed verdict and all evidence are retained. Seven saved baseline files
were restored after matching hashes. `--resume-after-speed-stop` continues
only from this pre-final boundary with the recovered baseline and the existing
recording regression; it cannot launch another speed43 trial. The collision
and component checks must still finish before any final seed is generated.

The serial owner is `experiments/run_full_evaluation44.py`. Read `progress.json`
before launching anything; do not create a second simulator queue.

1. Verify a new run of SFP dev_45006 with the fixed post-motion recording tail.
2. Complete six matched speed-development trials. Select speed43 only if all
   six pass independent checks and each connector's median simulated action
   time improves by at least 10 percent. Otherwise restore the saved baseline.
3. Rerun measurement negative controls, a fresh deliberate-miss physical
   collision positive control, and four development component-removal trials.
4. Freeze source/model hashes. Only then generate the predefined fresh seeds.
5. Execute 50 SC and 50 SFP full-policy trials across balanced mounts/slots.
6. Execute the two component removals on a balanced 20-case subset per connector
   (80 more trials), selected before any final outcome.
7. Report separate policy/all-attempt denominators, Wilson intervals, successful
   and failed completion times, forces, coverage, failure causes and paired
   exact tests with Holm correction.

All physical criteria remain unchanged, including independent 1-second terminal
dwell and the 20N measured raw-force ceiling. The new coverage-only recording
tail observes timestamps without consulting successful geometry. Missing data
still causes a failed verification. The original insufficient-dwell result and
the perception rejection remain in the historical replay ledger.

`final_trials_started` must stay false until all development gates finish.
The primary and ablation summaries are kept separate. Every completed trial
is exported with trace, selected camera frames, source hashes and checksums;
the missing old adapted checkpoints are never represented as recovered binaries.
Do not modify policy sources once `source_freeze.json` exists. Inspect a stopped
job before any resume, and never tune using opened final cases. The setup-only
resume option applies solely while waiting for the first regression and before
any development outcome has been ingested.

The larger study will take many hours on the current laptop. A safe boundary
stops the job below 8GiB free space or after three consecutive environment
failures; unattempted cases remain unattempted. Other WSL projects are untouched.

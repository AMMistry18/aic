# SC/SFP last-inch insertion manuscript

`main.tex` is the new empirical draft. `MANUSCRIPT_DRAFT.md` and the dated
earlier drafts are preserved. `main.pdf` is the compiled review copy, not a
submission or a completed robustness study.

The project follows the user-supplied
[empirical-paper-writer skill](https://github.com/yunshenwuchuxun/latex-paper-skills/blob/main/.codex/skills/empirical-paper-writer/SKILL.md).
The four-page request overrides its longer-paper and citation-count defaults.
The author instructions' linked `ieeeconf.cls` is included unmodified.

## Files and evidence

- `plan/20261112-writing-contract.md`: authorized scope, outline and skill adaptations.
- `brief/evidence-matrix.csv`: claims, verified/planned state and boundaries.
- `notes/design/`: method components, baselines and experiment matrix.
- `issues/20261112-sc-sfp-draft.csv`: writing and QA checklist.
- `notes/citation_verification.csv`: checked primary sources and claim scope.
- `results/recovery_snapshot_20261112.json`: immutable partial replay snapshot.
- `results/recovery_trials_20261112.csv`: per-attempt data, including failure.
- `results/recovery_macros.tex`: tables generated from that snapshot.
- `figures/provenance.json`: hashes of actual camera pixels and evaluator traces.
- `notes/reviewer_report.md`: remaining scientific and presentation issues.

The frozen manuscript snapshot contains six verified SC policy trials and five
verified SFP policy trials out of six, plus one SFP bootstrap failure. A separate
aborted duplicate setup is retained in the snapshot. Ongoing live replay results
must not silently change this version of the manuscript. Create a new dated
snapshot and update abstract, results, discussion and claims together when the
full cohort is ready.

The original absent adapted model binaries have not been recovered. Recovered
models are newly trained revisions, preserved outside `.artifacts` under
`experiments/recovery_20261112/checkpoints/`. Historical and new counts are not
pooled. Earlier SC component ablations use a different revision and support only
their declared development comparison.

## Build

From the repository root, with Python and a TeX installation:

```powershell
python paper/build_research_contract.py
python paper/build_figures.py
paper/writing_support/bin/tectonic.exe --keep-logs --keep-intermediates paper/main.tex
python paper/check_draft.py
```

The research-contract generator preserves an existing snapshot. It does not
refresh numbers from live progress when that snapshot already exists. Figures
are derived from executed records. Their PDF/PNG rendering is a visualization
of evaluator data, not a simulated policy input.

## Before a submission version

Finish and inspect the frozen replay, resolve perception/initialization failures,
qualify the speed choice, execute fresh cases and current matched ablations, and
fill the explicitly pending result paragraph. Add the final authors and
affiliations, check the double-anonymous version and repository links, make the
reproducibility archive durable, and complete any venue-required disclosure.
Simulation-only eligibility is accepted from the user's collection-specific
guidance; no hardware result is implied.

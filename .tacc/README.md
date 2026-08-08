# TACC perception jobs

These jobs collect, validate, and train the pose models used by the current
insertion system. Run every command from the repository root so relative paths
resolve consistently.

| Directory | Purpose |
| --- | --- |
| `sc_mouth/` | Physical SC mouth data collection, training, and batch pipelines |
| `sc_plug/` | SC plug data collection and training |
| `sc_port/` | Legacy SC port-target collection and crop-refinement evaluation |
| `sfp_plug/` | SFP plug data collection, resume preparation, and training |

Each target directory uses consistent names:

- `collect.slurm` collects labelled simulator data.
- `train.slurm` trains the corresponding pose model.
- Additional scripts describe their narrower pipeline or evaluation role.

The committed jobs contain TACC-specific account and `/work2` paths. Review
those values before submitting from a different allocation or checkout.

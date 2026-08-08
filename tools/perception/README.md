# Perception tools

Shared trial generators:

- `generate_sc_trials.py` creates held-SC collection trials used by the SC
  mouth, plug, and port pipelines.
- `generate_sfp_trials.py` creates SFP plug collection trials.

Target-specific tools:

| Directory | Contents |
| --- | --- |
| `sc_mouth/` | Train and validate the physical SC mouth model |
| `sc_plug/` | Train, validate, and analyze the SC plug model |
| `sc_port/` | Legacy SC port training, evaluation, filtering, and sanity checks |
| `sfp_plug/` | Train and evaluate the SFP plug model, including macOS setup |

The deployed perception implementation lives in
`aic_example_policies/aic_example_policies/ros/`; these files prepare and
evaluate its model weights but are not imported by the runtime.

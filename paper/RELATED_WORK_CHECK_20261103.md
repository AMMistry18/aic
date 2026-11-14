# Primary-source check, November 3, 2026

This is a scope check, not an exhaustive novelty search.

| Primary source | Established contribution | Consequence for our claims |
|---|---|---|
| [Kim and Rodriguez, Active Extrinsic Contact Sensing (ICRA 2022)](https://arxiv.org/abs/2110.03555), [IEEE record](https://ieeexplore.ieee.org/abstract/document/9812017) | Active tactile estimation/control of external contact, followed by insertion using contact estimates | Active contact estimation or using contact as a low-dimensional control input is not itself new |
| [SyPAS, authors' project page](https://sypas2026.github.io/) | Synthetic socket perception, refinement/tracking, visual servoing and compliant connector insertion, with real-robot evaluation | Synthetic learning, visual correction and compliant connector insertion are not themselves new; our current simulation-only cycle times do not support a speed advantage |

Our prospective claim is narrower: connector-specific measurement allocation
for SC and SFP last-inch insertion, with explicit unsupported-pose assumptions,
bounded own-robot correction, qualified contact evidence and independent full-rate
force/geometry verification. This is an engineering contribution hypothesis.
November 5 update: the SFP controlled-excitation check is deployed and has
prospective independently verified insertion pilots, including V38's four
selected exposure-repair cases. Broad same-version qualification and matched
current-method ablations remain incomplete. This supports an implemented
mechanism, not a causal improvement or population-robustness claim.

Needed evidence: same frozen perception/control implementation across varied
target identities and aligned handoffs; paired removal of the axial coordinate
correction and lateral tracking; noise/missing-contact/obstruction controls;
per-connector success intervals, times, force peaks and preserved failure causes.
No state-of-the-art, hardware-transfer or arbitrary-orientation claim is supported.

The original team article initially could not be retrieved through the web
tool. This was resolved through its [GitHub Markdown source](https://github.com/SatyaA27/intrinsic-ai-for-industry-challenge-team-tar-2-sol/blob/main/index.md),
blob `6e0483ce24854aaa2aea6b28962ee0270ca31360`. Its full method was read and
compared against local code and actual runtime checkpoint hashes. See
`ORIGINAL_PIPELINE_TRACEABILITY.md` and `.artifacts/repair36/checkpoint_lineage.json`.

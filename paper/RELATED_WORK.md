# Literature and submission notes

2026-10-28 update: [NOVELTY_DECISION.md](NOVELTY_DECISION.md) compares the primary
sources for Insert-One, FORGE, ContactDP, SyPAS, keypoint-EKF visual servoing,
Active Extrinsic Contact Sensing, Touch2Insert and assembly dual control. The
selected hypothesis is geometry-checked selective visual refinement followed
by clearance-conditioned reacquisition/contact recovery. Its novelty and
end-to-end benefit remain to be established; vision-force fusion, an EKF,
synthetic training and active probing are not sufficient standalone claims.

Checked 2026-10-24; primary sources only. These are a positioning map, not a
claim of an exhaustive novelty search. Read full papers before final related-work
comparisons; abstracts do not establish relative performance.

| Work | Why it matters here | Boundary for our paper |
|---|---|---|
| [Factory, RSS 2022](https://arxiv.org/abs/2205.03532) | Contact simulation for assembly | A simulator score is not a physics-validity argument |
| [IndustReal, RSS 2023](https://arxiv.org/abs/2305.17110) | Contact-rich simulation and real-robot transfer | We have no equivalent hardware transfer evidence |
| [Insert-One, IROS 2024](https://www.merl.com/publications/TR2024-137) | Visual tracking plus impedance control; uncertain in-hand pose | Vision-force fusion and measured grasp handling are not new by themselves |
| [FORGE](https://arxiv.org/abs/2408.04587) | Force-aware policies and pose uncertainty; revised January 2025 | Compare force-constrained success, not success at unconstrained force |
| [Model-Based Connector Mating, CASE 2025](https://arxiv.org/abs/2503.09409) | Industrial connector mating with auditable robot programs and multimodal optimization | Our deterministic controller is an engineering choice, not proof against learning |
| [ContactDP, September 2026 preprint](https://arxiv.org/abs/2609.23800) | RGB, tactile, force sensing and hybrid control for tight connectors | Avoid a broad claim of novelty for closed-loop multimodal insertion |
| [Structured-compliance connector assembly](https://arxiv.org/abs/2307.15589) | Mechanical compliance and insertion speed | Runtime must include task scope and hardware/control differences |

The proposed differentiator is a reproducible, insertion-oriented failure
analysis connecting correlated image bias, multiview geometric accuracy,
geometry-normalized alignment, bounded recovery, and independently checked
success. Whether this is sufficient as a new practical contribution depends on
the E2/E3 evidence; it is not established by the present records alone.

## Venue

- [Author instructions](https://www.ieee-ras.org/publications/ra-p/information-for-authors/):
  4-page body including figures/tables, optional 1-page supplementary description,
  up to 1-page references; IEEE double-column format and double-anonymous review.
- [Special collection](https://www.ieee-ras.org/publications/ra-p/special-collections/):
  Dexterous Manipulation and AI Advancements in Electronics Manufacturing;
  December 1, 2026 submission deadline.
- General author instructions require real-world demonstrations. The user's
  supplied brief reports collection-specific September 30 guidance allowing
  expanded simulation without mandatory hardware. That correspondence is not in
  this checkout and has not been independently authenticated. Preserve it and
  clarify its applicability before submission; continue simulation preparation.
- Add source/data descriptions and companion plain-text descriptions for
  supplementary attachments; check licensing, author anonymity, and the IEEE
  generative-AI disclosure rules before final packaging.

## Original system narrative

The [team article](https://satyaa27.github.io/intrinsic-ai-for-industry-challenge-team-tar-2-sol/)
was retrieved successfully with HTTP on this host after the browsing tool failed.
An adjacent source checkout exists at
`C:/Users/anshu/College/Intrinsic/Article/vision-force-cable-insertion/index.md`.
Its perception claims were cross-checked against the committed JSON records.
The article's survey result (96 geometric scenarios) is distinct from insertion
success; it is not an end-to-end hardware result. The learned insertion branch
was retired with unresolved parity/transfer questions; no controlled head-to-head
comparison against the deterministic controller is presently established.

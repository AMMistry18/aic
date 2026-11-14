# Insertion event deployment policy

Any fresh `/scoring/insertion_event` is a successful insertion, even when the
event names a different cable or port than the current Flowstate task.

Do not restore wrong-port rejection in the AIC model. Flowstate process logic
owns task and port bookkeeping; the model must not abort a physically completed
insertion because the scoring event's identifiers differ from the request.

This policy applies to SFP and SC insertion. SC's default
`SC_STRICT_PORT_EVENT=false` behavior and the SFP controller's fresh-event
acceptance must remain aligned with it.

## Research evaluation boundary (2026-10-24)

The rule above defines deployment event acceptance. It must not be used as a
publication's physical-seating verifier. Record the requested cable and port,
the actual event identifiers, and independent geometry/contact evidence as
separate fields. A wrong-target event cannot satisfy requested-target success.
The Gazebo CablePlugin can make cable connections static after a trigger;
stability produced by that fixture is not evidence of contact-rich insertion.
The proposed evaluator-only verifier is `experiments/verification.py`; its
geometry thresholds and recorder still require validation against the pinned
model revision. Deployment behavior has not been changed by the research tools.

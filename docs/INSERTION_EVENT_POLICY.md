# Insertion event deployment policy

Any fresh `/scoring/insertion_event` is a successful insertion, even when the
event names a different cable or port than the current Flowstate task.

Do not restore wrong-port rejection in the AIC model. Flowstate process logic
owns task and port bookkeeping; the model must not abort a physically completed
insertion because the scoring event's identifiers differ from the request.

This policy applies to SFP and SC insertion. SC's default
`SC_STRICT_PORT_EVENT=false` behavior and the SFP controller's fresh-event
acceptance must remain aligned with it.

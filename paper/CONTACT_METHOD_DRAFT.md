# Conditional last-inch SC/SFP method — unqualified working text

This is methods text for the required combined paper, not a claim of robust
performance. Physical outcomes and source revisions are recorded separately in
`REPAIR_20261102.md`. No held-out insertion evaluation has run.

The robot receives a connector already grasped and angularly aligned near its
requested port. A rigid attachment is prescribed by the experiment, but its
transform is not supplied to the policy. RGB observations estimate that
transform. This isolates translation correction and compliant insertion from
pickup, transport and general angular recovery. The public level fixture
family used for SFP is an additional declared workcell prior. Unknown fixture
tilt or arbitrary initial angular error falls outside this evaluation.

The original learned plug and port models propose multiview correspondences.
Keypoint confidences weight geometric residuals, without being interpreted as
calibrated pose uncertainty. SC keeps the original plug checkpoint and an
adapted mouth checkpoint; SFP adapters start from the original NIC and plug
checkpoints on separate rendered training scenes. Source meshes and materials
are retained. Simulator labels are used only offline for adaptation. At run
time, calibrated camera intrinsics, image-exposure extrinsics and own-robot
kinematics are used; live simulator object and target transforms are unavailable
to the policy. Runtime input age and intercamera synchronization gates remain
in force. Stationary processing uses exposure-time TCP and separately checks
that the robot did not move while computing.

For SC, learned proposals initialize occlusion-aware CAD mouth/body image
fits. Translation validation distinguishes lateral control evidence from a
fully certified visual pose. The original 3D gate is preserved. A provisional
lateral candidate requires agreement from another exposure and a subsequent
axial contact measurement; it cannot certify depth on its own. This avoids
turning one accepted coordinate component into an unsupported full-pose claim.

For SFP, the two entrances on each NIC provide a public 23.2 mm pitch and
eight rectangular rim corners. Stereo-supported candidates are associated with
the held plug inside a bounded last-inch lateral/axial envelope. Single-view
depth ghosts are excluded. Directional RGB edges refine the public dual-rim
model, excluding the held plug footprint. The level fixture family constrains
unobservable tilt. Plug translation is then estimated from learned keypoints.
The controller preserves its initial physical wrist orientation rather than
rotating the robot using an inaccurate learned grasp angle.

Contact measures relative connector-to-port coordinates, not separate global
plug and port poses. SC provides a terminal shoulder near 15.64 mm. Slow
load/unload/reload requires continuing commanded advance, stationary observed
robot position, increasing positive compression, measured release, and two
consistent stops. SFP has no equivalent rigid stop at the nominal 45.8 mm
seat in the source collision model. Instead, opposed small lateral offsets
contact the entrance lip near a public 53.5 micrometre depth datum. Consistent
opposed lip measurements register the axial relative coordinate. At shallow
engagement, opposed clipped side-wall plateaus and statistically distinguishable
wrist reactions register each lateral coordinate. Free motion and unobservable
contact cause abstention. These measurements never use simulator contact flags.

Positive compression is the negative port-axis projection of tared wrist
force. A development error using the opposite sign was detected through
recorded physical probes; all affected revisions remain separate. Persistent
bounded setpoints then advance into the connector. Own-TCP lateral PI reduces
tracking bias, freezes during contact probes, and has bounded force authority.
The axial trajectory supplies enough lead to overcome declared damping and
brakes before the seat. Instantaneous wrist guards and action budgets remain
active in all proposed ablations.

An independent process checks actual depth, full angular/lateral error, stable
completion, contact-stream health, penetration, and every-physics-step wrist
force. Policy completion alone is insufficient. SC requires actual seating
contact; SFP's nominal seat is a clearance region, requiring a qualified
collision positive control and valid contact instrumentation. Neither result
demonstrates electrical engagement, latch operation or stability after grasp
release. Physics fidelity and the rigid grasp must accompany the claims.

Generic confidence weighting, CAD fitting, vision/force fusion and active
contact localization are established. The candidate engineering contribution
is the explicit allocation of visual and contact-observable error components
to the distinct SC shoulder and SFP entrance geometries, with abstention and
independent physical verification. Matched removal of axial registration,
lateral registration and visual refinement must establish their effects on
both connectors before this becomes a supported contribution.

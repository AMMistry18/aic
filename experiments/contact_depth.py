"""Explicit engine-specific interpretation; raw contact packets stay unchanged."""
import math

SIGNED_DISTANCE='signed_distance_negative_penetration'
PENETRATION='positive_penetration'

def penetration_m(raw,convention):
    value=float(raw)
    if not math.isfinite(value):raise ValueError('Nonfinite contact distance')
    if convention==SIGNED_DISTANCE:return max(0.,-value)
    if convention==PENETRATION:return max(0.,value)
    raise ValueError(f'Unaudited depth convention: {convention}')

def touching(contact,convention):
    # Positive Bullet separation is a cached manifold point, not proof of touch.
    depths=contact['depths']
    if convention==SIGNED_DISTANCE:return any(float(d)<=0 for d in depths)
    if convention==PENETRATION:return any(float(d)>=0 for d in depths)
    raise ValueError(f'Unaudited depth convention: {convention}')


def active_constraint_impulses(contact,runtime):
    """Read the audited 8.3 scalar impulse, never interpret xyz as a force.

    A positive-distance cached manifold alone is not contact evidence. A
    nonzero applied normal impulse establishes an active solver constraint.
    Only the exact duplicated/opposite scalar encoding audited in 8.3 is read.
    """
    if runtime.get('gz_physics_version')!='8.3.0' or not runtime.get('plugin_sha256'):
        return []
    if not runtime.get('contact_wrench_semantics','').startswith('not a force vector: 8.3.0 reports scalar impulse'):
        return []
    depths=contact.get('depths',[]);wrenches=contact.get('wrenches',[])
    if len(depths)!=len(wrenches):return []
    active=[]
    for depth,wrench in zip(depths,wrenches):
        try:
            first=[float(wrench['body_1_wrench']['force'][k]) for k in ('x','y','z')]
            second=[float(wrench['body_2_wrench']['force'][k]) for k in ('x','y','z')]
            distance=float(depth)
        except (KeyError,TypeError,ValueError):continue
        if not all(math.isfinite(v) for v in [distance,*first,*second]):continue
        if first[0]>0 and first==[first[0]]*3 and second==[-first[0]]*3:
            active.append({'raw_signed_distance_m':distance,'normal_impulse_Ns':first[0]})
    return active


def physical_contact(contact,runtime):
    return touching(contact,runtime['depth_convention']) or bool(active_constraint_impulses(contact,runtime))

def resolve_contract(spec,audit=None):
    data=spec.get('physics_contact_contract') or audit
    if not data:raise ValueError('Missing audited contact convention; legacy runs require --runtime-audit')
    if data.get('depth_convention') not in (SIGNED_DISTANCE,PENETRATION):raise ValueError('Missing audited depth convention')
    return data


def same_engine_contract(left,right):
    """A contact control from a different engine binary is not transferable."""
    keys=('plugin_sha256','gz_physics_version','depth_convention')
    return bool(left and right and all(left.get(k) and left.get(k)==right.get(k) for k in keys))

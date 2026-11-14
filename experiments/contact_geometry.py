"""Independent capsule/box intersection math, without engine contact fields."""
import numpy as np


def segment_box_distance(start,end,half_size):
    """Exact Euclidean distance between a segment and a centered solid AABB.

    Squared distance is a convex piecewise quadratic in segment parameter.
    Split at crossings of each box face and minimize each quadratic exactly.
    """
    start,end,half_size=(np.asarray(v,dtype=float) for v in (start,end,half_size))
    if any(v.shape!=(3,) or not np.isfinite(v).all() for v in (start,end,half_size)):
        raise ValueError('Expected three finite 3-vectors')
    if np.any(half_size<=0):raise ValueError('Box extents must be positive')
    direction=end-start;breaks=[0.,1.]
    for axis in range(3):
        if direction[axis]!=0:
            breaks.extend(float(t) for t in ((-half_size[axis]-start[axis])/direction[axis],
                (half_size[axis]-start[axis])/direction[axis]) if 0<t<1)
    breaks=sorted(set(breaks));candidates=list(breaks)
    for left,right in zip(breaks,breaks[1:]):
        point=start+direction*((left+right)/2)
        mask=np.abs(point)>half_size
        slope=direction[mask];offset=start[mask]-np.sign(point[mask])*half_size[mask]
        denom=float(slope@slope)
        if denom:candidates.append(float(np.clip(-(slope@offset)/denom,left,right)))
    return min(float(np.linalg.norm(np.maximum(np.abs(start+direction*t)-half_size,0))) for t in candidates)


def capsule_box_overlap_lower_bound(capsule_world,length,radius,box_world,size):
    """Geometric intersection witness, not the engine's manifold depth.

    SDF length is the cylindrical section along local Z. If the centerline
    intersects the box, radius is only a lower bound on separation translation;
    otherwise radius minus centerline distance is the overlap lower bound.
    https://sdformat.org/spec/1.12/geometry/
    """
    if not np.isfinite([length,radius]).all() or length<0 or radius<=0:
        raise ValueError('Invalid capsule dimensions')
    for value in (capsule_world,box_world):
        if np.asarray(value).shape!=(4,4) or not np.isfinite(value).all():raise ValueError('Invalid transform')
    local=np.linalg.inv(box_world)@capsule_world
    ends=[(local@np.array([0.,0.,z,1.]))[:3] for z in (-length/2,length/2)]
    distance=segment_box_distance(*ends,np.asarray(size)/2)
    return {'centerline_distance_m':distance,'overlap_lower_bound_m':max(0.,radius-distance),
        'centerline_intersects_box':distance<=1e-12}

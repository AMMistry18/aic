import numpy as np
import pytest
from experiments.contact_geometry import segment_box_distance,capsule_box_overlap_lower_bound


def test_segment_box_distance_crossings_and_degenerate_segments():
    assert segment_box_distance([-2,0,0],[2,0,0],[1,1,1])==0
    assert segment_box_distance([-2,2,0],[2,2,0],[1,1,1])==1
    assert segment_box_distance([2,2,0],[2,2,0],[1,1,1])==pytest.approx(np.sqrt(2))
    # Closest point is strictly inside the segment, near a box corner.
    assert segment_box_distance([0,3,0],[3,0,0],[1,1,1])==pytest.approx(1/np.sqrt(2))


def test_capsule_rotated_box_and_interior_lower_bound():
    capsule=np.eye(4);capsule[0,3]=1.2
    result=capsule_box_overlap_lower_bound(capsule,1,.3,np.eye(4),[2,2,2])
    assert result['overlap_lower_bound_m']==pytest.approx(.1)
    R=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    motion=np.eye(4);motion[:3,:3]=R;motion[:3,3]=[10,20,30]
    moved=capsule_box_overlap_lower_bound(motion@capsule,1,.3,motion,[2,2,2])
    assert moved['overlap_lower_bound_m']==pytest.approx(.1)
    inside=capsule_box_overlap_lower_bound(np.eye(4),1,.3,np.eye(4),[2,2,2])
    assert inside['centerline_intersects_box']
    assert inside['overlap_lower_bound_m']==pytest.approx(.3)


def test_invalid_geometry_fails_closed():
    with pytest.raises(ValueError):segment_box_distance([np.nan,0,0],[1,0,0],[1,1,1])
    with pytest.raises(ValueError):segment_box_distance([0,0,0],[1,0,0],[0,1,1])

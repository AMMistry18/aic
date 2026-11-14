"""Synthetic sensor controls only. This candidate is not in the live policy."""
import numpy as np
from experiments.pulse_contact36_candidate import fixed_pair,fixed_triplet


def observation(seed,reaction):
    rng=np.random.default_rng(seed)
    low=np.c_[np.arange(21)*.02,np.zeros(21),rng.normal(0,.2,21)]
    high=np.c_[.6+np.arange(21)*.02,np.zeros(21),reaction+rng.normal(0,.2,21)]
    return low,high


def check(low,high,**kwargs):
    return fixed_pair(low,high,low_end=.4,high_end=1.,kinematic_blocked=True,**kwargs)


def test_fixed_noise_negative_and_positive_controls():
    negatives=[];positives=[]
    for seed in range(7000,7032):
        negatives.append(check(*observation(seed,0.))['accepted'])
        positives.append(check(*observation(seed,1.))['accepted'])
    assert sum(negatives)==0
    assert all(positives)


def test_sensor_duplicates_stale_data_motion_and_no_block_reject():
    low,high=observation(7000,1.)
    dup=high.copy();dup[3,0]=dup[2,0]
    assert not check(low,dup)['accepted']
    assert not fixed_pair(low,high,low_end=.4,high_end=2.,kinematic_blocked=True)['accepted']
    moved=high.copy();moved[:,1]=np.linspace(0,.0001,len(high))
    assert not check(low,moved)['accepted']
    shifted=high.copy();shifted[:,1]=.0001
    assert not check(low,shifted)['accepted']
    assert not fixed_pair(low,high,low_end=.4,high_end=1.,kinematic_blocked=False)['accepted']


def test_fixed_reversible_excitation_rejects_monotonic_drift():
    for seed in range(7000,7032):
        low,high=observation(seed,1.)
        rng=np.random.default_rng(seed+100)
        released=np.c_[1.2+np.arange(21)*.02,np.zeros(21),rng.normal(0,.2,21)]
        kwargs=dict(low_end=.4,high_end=1.,release_end=1.6,kinematic_blocked=True)
        assert fixed_triplet(low,high,released,**kwargs)['accepted']
        drift=released.copy();drift[:,2]+=2.
        assert not fixed_triplet(low,high,drift,**kwargs)['accepted']
        no_reaction=high.copy();no_reaction[:,2]-=1.
        assert not fixed_triplet(low,no_reaction,released,**kwargs)['accepted']

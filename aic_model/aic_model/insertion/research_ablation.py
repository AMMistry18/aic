"""Prespecified component removals; never changes the independent verifier."""
import numpy as np

MODES=('full','no_axial_correction','no_lateral_tracking')


def mode(policy):
    value=getattr(policy,'_research_ablation','full')
    if value not in MODES:raise ValueError('Unknown research ablation')
    return value


def axial_correction(policy,measured):
    if not np.isfinite(measured):raise ValueError('Nonfinite measured correction')
    return 0. if mode(policy)=='no_axial_correction' else float(measured)


def lateral_disabled(policy):return mode(policy)=='no_lateral_tracking'

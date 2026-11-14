"""Separate bounded search intent from stationary lip measurement readiness.

This only triggers one reversible force test; it never accepts contact.
Inputs are own tip estimates/commands and their recorded simulation times.
"""
import numpy as np


def stationary_search_candidate(samples):
    if not samples:return None
    history=np.asarray(samples,dtype=float)
    if history.ndim!=2 or history.shape[1]!=4 or not np.isfinite(history).all():return None
    if np.any(np.diff(history[:,0])<=0):return None
    rows=history[history[:,0]>=history[-1,0]-.83]
    if len(rows)<20 or rows[-1,0]-rows[0,0]<.77:return None
    gaps=np.diff(rows[:,0]);advance=float(rows[-1,2]-rows[0,2])
    search_advance=float(np.max(history[:,2])-history[0,2])
    travel=float(np.ptp(rows[:,1]));lead=float(np.median(rows[-8:,2]-rows[-8:,1]))
    # Once an elastic command saturates against a lip, its recent advance is
    # zero. A prior sampling gap must not erase the already executed search.
    # Require the same fresh stationary window and elastic lead, then let the
    # separate fixed 0/1/0 N test establish reaction/release without retries.
    if gaps.max()>.05 or search_advance<.0001 or travel>.000025 or lead<.00015:return None
    return dict(command_advance_m=advance,search_command_advance_m=search_advance,
                observed_travel_m=travel,elastic_lead_m=lead,
                trigger_scope='bounded search history plus current stationary window; force proof still required')

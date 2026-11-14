"""Undeployed fixed-look force-excitation check; observations only, no truth.

Rows: actual wrist sensor time, own estimated tip depth, positive compression.
The caller must apply one fixed low/high excitation pair, enforce workspace and
force guards, and establish kinematic blocking separately. Statistical margins
here are engineering checks, not a calibrated contact probability.
"""
import numpy as np


def plateau(rows,now,*,window_s=.4):
    rows=np.asarray(rows,dtype=float)
    if rows.ndim!=2 or rows.shape[1]!=3 or len(rows)<10 or not np.isfinite(rows).all():
        return None,'missing_or_nonfinite_samples'
    t=rows[:,0];gaps=np.diff(t)
    if np.any(gaps<=0) or gaps.max()>.05 or now-t[-1]>.05 or now<t[-1]:
        return None,'stale_duplicate_or_gapped_sensor'
    if t[-1]-t[0]<window_s-.06:return None,'insufficient_fixed_window'
    if np.ptp(rows[:,1])>.000025:return None,'tip_not_stationary'
    # Known source noise is 0.2 N/axis. Never report lower uncertainty merely
    # because a small batch happens to have low empirical sample variance.
    sem=max(.2,float(rows[:,2].std(ddof=1)))/np.sqrt(len(rows))
    return dict(mean_N=float(rows[:,2].mean()),sem_N=float(sem),samples=len(rows),
                depth_m=float(np.median(rows[:,1])),span_s=float(t[-1]-t[0])),None


def fixed_pair(low,high,*,low_end,high_end,kinematic_blocked,applied_pulse_N=1.):
    result=dict(accepted=False,simulator_truth_used=False,physical_qualification=False,
                fixed_pair_only=True,minimum_contrast_N=.025,noise_margin_multiplier=3.,
                applied_pulse_N=applied_pulse_N)
    if not kinematic_blocked:return {**result,'reason':'no_kinematic_block'}
    if applied_pulse_N!=1.:return {**result,'reason':'unknown_excitation_protocol'}
    first,reason=plateau(low,low_end)
    if reason:return {**result,'reason':'low:'+reason}
    second,reason=plateau(high,high_end)
    if reason:return {**result,'reason':'high:'+reason}
    if high[0][0]<=low[-1][0]:return {**result,'reason':'windows_overlap'}
    # Loading must preserve the same apparent contact location; substantial
    # motion is an unresolved/free-space observation, not a contact datum.
    if abs(second['depth_m']-first['depth_m'])>.000025:
        return {**result,'reason':'contact_location_moved'}
    contrast=second['mean_N']-first['mean_N']
    margin=3.*np.hypot(first['sem_N'],second['sem_N'])
    accepted=contrast>=max(.025,margin)
    return {**result,'accepted':bool(accepted),'reason':'force_excitation_observed' if accepted else 'reaction_not_resolved',
            'low':first,'high':second,'contrast_N':float(contrast),'margin_3sem_N':float(margin)}


def fixed_triplet(low,high,released,*,low_end,high_end,release_end,kinematic_blocked):
    """Require the reaction to decrease again after the one fixed excitation.

    Returning to baseline rejects monotonic force drift; no optional retries or
    selecting a favourable window. The physical caller is not implemented yet.
    """
    result=fixed_pair(low,high,low_end=low_end,high_end=high_end,kinematic_blocked=kinematic_blocked)
    result['fixed_triplet_only']=True
    if not result['accepted']:return result
    last,reason=plateau(released,release_end)
    if reason:return {**result,'accepted':False,'reason':'release:'+reason}
    if released[0][0]<=high[-1][0]:return {**result,'accepted':False,'reason':'release_windows_overlap'}
    first=result['low'];second=result['high']
    if abs(last['depth_m']-first['depth_m'])>.000025:
        return {**result,'accepted':False,'reason':'release_contact_location_moved'}
    fall=second['mean_N']-last['mean_N'];margin=3*np.hypot(second['sem_N'],last['sem_N'])
    recovery=abs(last['mean_N']-first['mean_N']);baseline_margin=3*np.hypot(first['sem_N'],last['sem_N'])
    accepted=fall>=max(.025,margin) and recovery<=max(.025,baseline_margin)
    return {**result,'accepted':bool(accepted),'reason':'reversible_force_excitation_observed' if accepted else 'reaction_not_reversible',
            'released':last,'force_fall_N':float(fall),'force_fall_margin_3sem_N':float(margin),
            'baseline_change_N':float(recovery),'baseline_margin_3sem_N':float(baseline_margin)}

"""Coverage-only terminal recording extension, without inspecting seat geometry."""
import math


class RecordingTail:
    def __init__(self,minimum_sim_s=1.5,maximum_gap_s=.03):
        if minimum_sim_s<=0 or maximum_gap_s<=0:raise ValueError('Positive coverage limits required')
        self.minimum=minimum_sim_s;self.gap=maximum_gap_s
        self.start=None;self.last=None;self.resets=0

    def observe(self,stamp):
        if not math.isfinite(stamp):raise ValueError('Finite pose stamp required')
        if self.last is not None and stamp<=self.last:return False
        if self.last is None or stamp-self.last>self.gap:
            self.start=stamp;self.resets+=int(self.last is not None)
        self.last=stamp
        return self.last-self.start>=self.minimum-1e-9

    def diagnostics(self):
        return dict(required_continuous_sim_s=self.minimum,max_gap_s=self.gap,
            observed_continuous_sim_s=0. if self.start is None else self.last-self.start,
            gap_resets=self.resets,geometry_used=False)

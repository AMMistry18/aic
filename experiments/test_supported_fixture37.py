"""Eager/lazy selector identity and numerical-fit call counts on pure fixtures."""
import importlib.util
import itertools
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('support_candidate',ROOT/'.artifacts/repair37/perception_candidate/sfp_fixture_support.py')
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)


def eager(cams,plug):
    rows=[]
    for count in range(2,len(cams)+1):
        for subset in itertools.combinations(cams,count):
            for picks in itertools.product(*(cams[c] for c in subset)):
                fit=candidate.fit_fixture(picks)
                if fit is None:continue
                for slot,offset in enumerate((np.zeros(3),candidate.SLOT1)):
                    p=np.array(fit['position'])+np.array(fit['rotation'])@offset
                    rows.append(dict(fit,position=p.tolist(),slot=slot,cameras=list(subset)))
    return candidate.select_candidate(rows,plug)


@pytest.mark.parametrize('order',list(itertools.permutations(['left','center','right'])))
@pytest.mark.parametrize('triple',['eligible','bad_fit','outside_capture'])
def test_exact_choice_matches_eager_priority_and_tie_order(monkeypatch,order,triple):
    cams={c:[dict(camera=c,index=i) for i in range(2)] for c in order}
    calls=[]
    def fit(picks):
        calls.append(picks)
        if len(picks)==3 and triple=='bad_fit':return None
        x=.01 if len(picks)==3 and triple=='outside_capture' else .0001*sum(p['index'] for p in picks)
        # Pairs can have lower residual/distance but cannot beat valid triples.
        return dict(position=[x,0,0],rotation=np.eye(3).tolist(),residual_px=.1 if len(picks)==2 else 1.)
    monkeypatch.setattr(candidate,'fit_fixture',fit)
    expected=eager(cams,np.array([0,0,-.01]));before=len(calls)
    measured,diag=candidate.fit_supported_fixture(cams,np.array([0,0,-.01]))
    assert measured==expected
    assert before==20 and diag['fixture_fit_calls']==(8 if triple=='eligible' else 20)


def test_missing_views_and_empty_detections_abstain(monkeypatch):
    monkeypatch.setattr(candidate,'fit_fixture',lambda _:None)
    with pytest.raises(ValueError):candidate.fit_supported_fixture({'left':[],'right':[]},np.array([0,0,-.01]))

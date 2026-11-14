"""Prepare an exactly ordered lazy-support search; do not change live sources."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.artifacts/repair37/perception_candidate';OUT.mkdir(parents=True,exist_ok=True)
source=ROOT/'aic_model/aic_model/insertion/sfp_fixture_refinement.py'
text=source.read_text()
helper='''def fit_supported_fixture(cams,plug):
    """Preserve maximum-view-support priority while skipping dominated fits.

    Within each support level the original camera/product/slot ordering and
    exact numerical fit are unchanged. Lower-support fits cannot win once an
    eligible higher-support candidate exists. No residual gate is changed.
    """
    fit_calls=0;levels=[]
    for count in range(len(cams),1,-1):
        candidates=[]
        for subset in itertools.combinations(cams,count):
            for picks in itertools.product(*(cams[c] for c in subset)):
                fit_calls+=1;fit=fit_fixture(picks)
                if fit is None:continue
                for slot,offset in enumerate((np.zeros(3),SLOT1)):
                    p=np.array(fit['position'])+np.array(fit['rotation'])@offset
                    candidates.append(dict(fit,position=p.tolist(),slot=slot,cameras=list(subset)))
        levels.append(count)
        try:seed=select_candidate(candidates,plug)
        except ValueError:continue
        return seed,dict(fixture_fit_calls=fit_calls,support_levels_evaluated=levels,
                         selected_view_support=count,ordering_equivalent_to_eager=True)
    raise ValueError('No stereo NIC candidate in last-inch capture envelope')


'''
assert 'def fit_supported_fixture(' not in text
candidate=text.replace('def refine_rim(views,seed,plug):',helper+'def refine_rim(views,seed,plug):',1)
start=candidate.index('    candidates=[]\n',candidate.index('def perceive('))
end=candidate.index('    plug,plug_error=',start)
candidate=candidate[:start]+'    seed,search=fit_supported_fixture(cams,measured.position_world)\n'+candidate[end:]
candidate=candidate.replace('policy._sfp_fixture_diagnostic=dict(seed=seed,rim=rim,','policy._sfp_fixture_diagnostic=dict(seed=seed,rim=rim,fixture_search=search,',1)
(OUT/'sfp_fixture_full.py').write_text(text)
(OUT/'sfp_fixture_support.py').write_text(candidate)
(OUT/'candidate.json').write_text(json.dumps(dict(installed=False,base_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    candidate_sha256=hashlib.sha256((OUT/'sfp_fixture_support.py').read_bytes()).hexdigest(),
    scope='same candidates and fit gates; stop evaluating support levels that cannot win'),indent=2)+'\n')
print(OUT)

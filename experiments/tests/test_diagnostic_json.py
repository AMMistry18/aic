import hashlib
import json
import numpy as np
from experiments.diagnostic_json import compact_diagnostics

def test_full_resolution_masks_do_not_expand_into_json_or_remain_retained():
    mask=np.ones((1024,1152),dtype=np.uint8)
    records=[{'camera':'left','bores':[{'valid_mask':mask,'center':np.array([20.,30.])}]} for _ in range(50)]
    bounded=compact_diagnostics(records)
    encoded=json.dumps(bounded)
    assert len(encoded)<20000
    summary=bounded[0]['bores'][0]['valid_mask']
    assert summary['shape']==[1024,1152]
    assert summary['nonzero_pixels']==1024*1152
    assert summary['sha256']==hashlib.sha256(mask.tobytes()).hexdigest()
    assert bounded[0]['bores'][0]['center']==[20.,30.]
    mask[:]=0
    assert summary['nonzero_pixels']==1024*1152

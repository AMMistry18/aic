"""Bound diagnostic image arrays before retaining or serializing them."""
import hashlib
import numpy as np

def compact_diagnostics(value):
    if isinstance(value,np.ndarray):
        if value.size<=256:return value.tolist()
        return {'shape':list(value.shape),'dtype':str(value.dtype),'nonzero_pixels':int(np.count_nonzero(value)),
            'sha256':hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {key:compact_diagnostics(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):return [compact_diagnostics(item) for item in value]
    return value

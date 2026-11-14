"""Reuse immutable checkpoint objects for future copied-workspace snapshots."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];DEV=ROOT/'.artifacts/dev46_workspace'


def main():
    objects=DEV/'.artifacts/frozen_checkpoint_objects';objects.mkdir(exist_ok=True)
    files=list((DEV/'aic_example_policies/aic_example_policies/ros/weights').glob('*.pt'))
    files+=list((DEV/'experiments/recovery_20261112/checkpoints').glob('*/best.pt'))
    rows=[]
    for source in files:
        digest=hashlib.sha256(source.read_bytes()).hexdigest();target=objects/(digest+'.pt')
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise RuntimeError('Checkpoint object changed')
        else:target.hardlink_to(source)
        rows.append(dict(path=str(source.relative_to(DEV)),sha256=digest,bytes=source.stat().st_size))
    (ROOT/'experiments/runtime_qualification47_v2/checkpoint_object_reuse.json').write_text(json.dumps(dict(
        purpose='Future evidence snapshots hardlink immutable weights; no existing evidence deleted or weights changed',rows=rows),indent=2))
    print(json.dumps(dict(objects=len(rows),future_duplicate_bytes_avoided_per_full_snapshot=sum(r['bytes'] for r in rows))))


if __name__=='__main__':main()

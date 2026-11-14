"""Portable actual diagnostics and code versions; no model binaries or new claims."""
import hashlib
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'experiments/evidence/original_repair/perception_repair30';out.mkdir(exist_ok=True)
    references=['crop_resolution30','nativecrop_precision30_best','nativecrop_precision30_last','live_nativecrop_precision30',
        'foreground_precision30_last','foreground_anchor30_last','foreground_view_subsets30','foreground_stereo_phase30_last']
    for name in references:
        source=ROOT/'.artifacts/reference30'/name;destination=out/name;destination.mkdir(exist_ok=True)
        for file in source.glob('*.json'):shutil.copy2(file,destination/file.name)
        if (source/'source_snapshot').exists():shutil.copytree(source/'source_snapshot',destination/'source_snapshot',dirs_exist_ok=True)
    for name in ['sc_plug_nativecrop_precision30_v1','sc_plug_foreground_precision30_v1']:
        destination=out/'training'/name;destination.mkdir(parents=True,exist_ok=True);source=ROOT/'.artifacts/perception_data30/training'/name
        for file in ['training_plan.json','training_result.json','run/results.csv']:
            p=source/file
            if p.exists():shutil.copy2(p,destination/p.name)
    for name in ['nativecrop_mixed_v1','foreground_native6_v1']:
        destination=out/'datasets'/name;destination.mkdir(parents=True,exist_ok=True);source=ROOT/'.artifacts/perception_data30'/name
        for filename in ['manifest.json','dataset_SHA256SUMS.json']:shutil.copy2(source/filename,destination/filename)
    files=['audit_sc_crop_resolution30.py','audit_sc_plug_checkpoint30.py','audit_sc_foreground_refinement30.py','sc_foreground_refinement30.py',
        'prepare_nativecrop_training30.py','prepare_foreground_training30.py','train_perception_gpu30.py','precision_keypoint_loss.py',
        'analyze_sc_view_subsets30.py','audit_rigid_grasp_invariance30.py','inspect_grasp_image_pair30.py','replay_sc_live_screen30.py','replay_sc_live_screen30.sh',
        'configs/foreground_refinement30.json','configs/foreground_anchor30.json','configs/foreground_stereo_phase30.json']
    for name in files:
        destination=out/'latest_source'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'experiments'/name,destination)
    summary=dict(purpose='Offline DEVELOPMENT diagnostics, not insertion results; no candidate passed declared qualification',
        training_scenes=20,static_development_scenes=6,recorded_live_captures=9,rigid_pairs=4,
        original_weights_unchanged=True,new_plug_model_deployed=False,final_cases_used=False,
        nativecrop_best=json.loads((out/'nativecrop_precision30_best/diagnostic_scores.json').read_text()),
        foreground_primary=json.loads((out/'foreground_precision30_last/summary.json').read_text()),
        anchored_primary=json.loads((out/'foreground_anchor30_last/summary.json').read_text()),
        view_subsets=json.loads((out/'foreground_view_subsets30/summary.json').read_text()),
        stereo_phase=json.loads((out/'foreground_stereo_phase30_last/summary.json').read_text()))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'README.md').write_text('''# Perception repair diagnostics, 2026-10-30

These are actual offline development measurements, not physical insertions or
final reliability. Original checkpoint and physical-policy thresholds remain
unchanged. No candidate below qualified for deployment.

The same 20 training scenes and six static development scenes were used for
adaptation. Nine archived live captures and four before/after rigid-grasp pairs
probe transfer and consistency; these have been used for repair and are not a
held-out final set. Image-only fits were saved before scoring simulator truth.

Native-crop-matched training improved median static translation, but did not
repair rotation. Foreground refinement reached 0.641 deg static median angular
error. Fixed side-view phase averaging reached 0.306 deg median on recorded
captures, yet passed only 1/4 unchanged rigid-consistency checks. A lower median
does not establish task-level accuracy or robustness. See the unabridged JSONs.

Actual paired images show orange cable occlusion in the centre view. Its point
scores remain high; confidence is not calibrated pose uncertainty. The grayscale
foreground prototype also admits some gray background, visible in the preserved
diagnostic figure. Crop anchoring and simple view rejection did not solve all
failures. No numerical acceptance or physical scoring threshold was relaxed.

Original and intermediate foreground source versions are retained within each
diagnostic's source_snapshot. latest_source contains the final diagnostic code;
it must not be substituted for older versions when reproducing exact outputs.
Training plans/results record actual checkpoint hashes; model binaries are not
duplicated in this portable export. Dataset manifests and image/label hashes
preserve the training provenance. These results do not replicate the article's
older held-out split or its reported 0.273 mm result.
''',encoding='utf-8')
    print(json.dumps(dict(export=str(out),files=sum(p.is_file() for p in out.rglob('*')),new_model_deployed=False)))


if __name__=='__main__':main()

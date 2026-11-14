"""Export actual exposure-fix development results, keeping revisions separate."""
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/evidence/exposure40'


def wilson(k, n):
    z = 1.959963984540054
    den = 1 + z*z/n
    center = (k/n + z*z/(2*n))/den
    half = z*math.sqrt(k/n*(1-k/n)/n + z*z/(4*n*n))/den
    return [center-half, min(1., center+half)]


def main():
    source = ROOT / '.artifacts/repair40/pilot_completed.json'
    batch = json.loads(source.read_text())
    rows = []
    for r in batch['results']:
        audit = r['high_rate_wrist_audit']
        m = r.get('terminal_metrics', {})
        rows.append(dict(
            run=r['run'], verified_success=r['verified_success'],
            reason=r['reason'], wall_s=r['wall_s'], action_sim_s=r['sim_s'],
            raw_peak_tared_force_N=audit['peak_tared_force_N'],
            raw_p95_tared_force_N=audit['p95_tared_force_N'],
            complete_2000Hz_coverage=audit['complete_action_coverage'],
            samples_over_20N=audit['samples_over_ceiling'],
            depth_min_mm=m['depth_m']['min']*1000,
            depth_max_mm=m['depth_m']['max']*1000,
            lateral_max_mm=m['lateral_error_m']['max']*1000,
            angle_max_deg=m['angular_error_deg']['max'],
            penetration_max_mm=m['max_penetration_m']['max']*1000,
        ))
    k = sum(r['verified_success'] for r in rows)
    summary = dict(
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose='selected_development_repair_validation_not_population_reliability',
        source=str(source.relative_to(ROOT)),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        revision='V38_exposure40v1', completed=len(rows), successes=k,
        wilson_95_interval=wilson(k, len(rows)),
        confidence_interval_limitation='Selected reused development cases; interval is descriptive, not a held-out robustness estimate.',
        median_action_sim_s=statistics.median(r['action_sim_s'] for r in rows),
        median_wall_s=statistics.median(r['wall_s'] for r in rows),
        min_raw_peak_N=min(r['raw_peak_tared_force_N'] for r in rows),
        max_raw_peak_N=max(r['raw_peak_tared_force_N'] for r in rows),
        all_complete_2000Hz=all(r['complete_2000Hz_coverage'] for r in rows),
        rows=rows,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8', newline='\n')
    with (OUT/'trials.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()

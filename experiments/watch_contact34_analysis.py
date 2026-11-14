"""Refresh completed-case analysis while the existing frozen study runs.

Read-only with respect to policy, source freeze, cases and simulator. Writes
analysis artifacts only. Exits when the owning queue finishes or stops.
"""
import datetime
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.analyze_contact34_validation import main as analyze,OUT


def main():
    previous=None;deadline=time.monotonic()+12*3600
    while time.monotonic()<deadline:
        try:
            state=json.loads((OUT/'progress.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            time.sleep(5);continue
        signature=(state['status'],len(state['results']))
        if signature!=previous:
            analyze();previous=signature
            (OUT/'analysis_watcher.json').write_text(json.dumps(dict(
                updated_local=datetime.datetime.now().astimezone().isoformat(),
                completed=len(state['results']),study_status=state['status'],
                policy_or_simulator_mutations=False),indent=2)+'\n')
        if state['status'] not in ('awaiting_nominal_qualification','executing_development_case'):return
        time.sleep(30)
    raise TimeoutError('Analysis watcher reached 12-hour limit; owning evaluation is unchanged')


if __name__=='__main__':main()

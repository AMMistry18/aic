from pathlib import Path
from experiments.run_seeded_evaluation import SOURCES

def test_frozen_shell_programs_use_linux_line_endings():
    root=Path(__file__).resolve().parents[2]
    for name in SOURCES:
        if name.endswith('.sh'):
            data=(root/name).read_bytes()
            assert data.startswith(b'#!/usr/bin/env bash\n'),name
            assert b'\r' not in data,name

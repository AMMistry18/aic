"""Offline, timestamp-faithful replay of genuine archived Gazebo camera JPEGs.

No simulation is launched and no intermediate robot images are synthesized.
Camera frames use zero-order hold; evaluation telemetry is labelled separately.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import math
import pathlib
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = {
    'sc': 'sc_supported_damped27repeat3_force_tactile_no_registration',
    'sfp': 'sfp_supported_sfpdamped27repeat3_force_tactile_no_registration',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    if path.suffix == '.gz':
        return json.loads(gzip.decompress(path.read_bytes()))
    return json.loads(path.read_text())


def font(size):
    for path in ['C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if pathlib.Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def previous(times, t):
    return max(0, bisect.bisect_right(times, t) - 1)


def make(kind, output):
    name = RUNS[kind]
    source = ROOT / '.artifacts/lastinch_runs' / name / name
    verdict = load(source / 'verified.json')
    assert verdict['verified_success'], 'This successful nominal replay requires a passing verdict.'
    policy = load(source / 'policy.json')
    assert policy['ground_truth_policy_inputs'] is False
    truth = load(source / 'truth_trace.json.gz')
    truth_times = [x['sim_time_s'] for x in truth]
    wrench = load(source / 'wrench.json.gz')
    force_times = [x['header']['stamp']['sec'] + x['header']['stamp']['nanosec'] * 1e-9 for x in wrench]
    wa = load(source / 'wrench_analysis.json')
    baseline = np.array(wa['baseline_N'])
    forces = [float(np.linalg.norm(np.array(list(x['wrench']['force'][k] for k in 'xyz')) - baseline)) for x in wrench]
    trace = policy['trace']
    phase_times = [x['sim_s'] for x in trace]
    panels = [('target', None, 'Target camera'), ('board', None, 'Board camera')] if kind == 'sc' else [
        ('board', (260, 130, 660, 500), 'Board camera - fixed crop, enlarged'),
        ('board', None, 'Board camera - full frame')]
    cameras = {}
    provenance = []
    for camera in dict.fromkeys(x[0] for x in panels):
        files = sorted(source.glob(camera + '_*.jpg'), key=lambda p: float(p.stem[len(camera) + 1:]))
        assert len(files) > 2
        times = [float(p.stem[len(camera) + 1:]) for p in files]
        assert all(b > a for a, b in zip(times, times[1:]))
        cameras[camera] = (times, files)
        provenance += [{'path': str(p.relative_to(ROOT)).replace('\\', '/'), 'sim_time_s': t, 'sha256': digest(p)} for t, p in zip(times, files)]
    # Common interval only: never display a future image before its capture time.
    start = max([truth_times[0], force_times[0]] + [v[0][0] for v in cameras.values()])
    end = min([truth_times[-1], force_times[-1]] + [v[0][-1] for v in cameras.values()])
    fps = 10
    count = math.floor((end - start) * fps) + 1
    video = output / f'{kind}_insertion.mp4'
    ffmpeg = shutil.which('ffmpeg')
    ffprobe = shutil.which('ffprobe')
    assert ffmpeg and ffprobe
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '1280x780', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2',
               '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    cache = {}
    selected = []
    try:
        for frame in range(count):
            t = start + frame / fps
            canvas = Image.new('RGB', (1280, 780), '#111923')
            d = ImageDraw.Draw(canvas)
            d.text((20, 12), f'{kind.upper()} last-inch insertion | nominal repeat 3', font=font(28), fill='white')
            d.text((20, 51), 'Actual Gazebo frames | 1x simulation time | sparse capture (~1 Hz); images held between captures', font=font(20), fill='#b9cbdc')
            camera_entries = []
            for i, (camera, crop, label) in enumerate(panels):
                times, files = cameras[camera]
                idx = previous(times, t)
                key = (camera, idx, crop)
                if key not in cache:
                    img = Image.open(files[idx]).convert('RGB')
                    if crop:
                        img = img.crop(crop)
                    img.thumbnail((636, 478), Image.Resampling.LANCZOS)
                    cache[key] = img
                img = cache[key]
                x = i * 640
                canvas.paste(img, (x + (640 - img.width) // 2, 112 + (480 - img.height) // 2))
                d.text((x + 10, 88), label, font=font(19), fill='white')
                d.text((x + 10, 596), f'Capture t={times[idx]:.3f} s | held {t-times[idx]:.2f} s', font=font(19), fill='#b9cbdc')
                camera_entries.append({'camera': camera, 'capture_sim_s': times[idx], 'crop_xyxy': crop})
            row = truth[previous(truth_times, t)]
            wi = previous(force_times, t)
            phase = trace[previous(phase_times, t)]['phase'] if t >= phase_times[0] else 'pre-command observation'
            d.text((20, 632), f't={t:.2f} s | Phase: {phase}', font=font(21), fill='white')
            d.text((20, 665), f'Evaluator GT: depth {row["depth_m"]*1000:.3f} mm | lateral {row["lateral_error_m"]*1000:.3f} mm | angle {row["angular_error_deg"]:.3f} deg', font=font(21), fill='#a9d7f7')
            d.text((20, 698), f'Tared force: {forces[wi]:.2f} N | Run peak: {wa["peak_independent_tared_force_N"]:.2f} N | Final independent verdict: PASS', font=font(21), fill='#b3e9c6')
            d.text((20, 738), 'Prescribed rigid grasp. GT is evaluation only. SFP seat is geometric; latch/electrical function untested.' if kind == 'sfp' else 'Prescribed rigid grasp. GT is evaluation only. SC terminal contact and continuous pose dwell verified.', font=font(18), fill='#b9cbdc')
            encoder.stdin.write(canvas.tobytes())
            if frame in [0, count // 2, count - 1]:
                canvas.save(output / f'{kind}_preview_{frame:04d}.jpg')
            selected.append({'video_frame': frame, 'sim_time_s': t, 'cameras': camera_entries,
                             'truth_sim_s': row['sim_time_s'], 'force_sim_s': force_times[wi]})
        encoder.stdin.close()
        error = encoder.stderr.read().decode(errors='replace')
        assert encoder.wait() == 0, error
    except BaseException:
        encoder.kill()
        encoder.wait()
        raise
    probe = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-count_frames', '-show_streams', '-show_format', '-of', 'json', str(video)]))
    stream = probe['streams'][0]
    assert stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p'
    assert int(stream['nb_read_frames']) == count
    for position, label in [(0, 'start'), (count//2/fps, 'middle'), ((count-1)/fps, 'end')]:
        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-ss', str(position), '-i', str(video), '-frames:v', '1', str(output/f'{kind}_decoded_{label}.jpg')], check=True)
    metadata = {
        'run': name, 'connector': kind, 'video_sha256': digest(video), 'render_script_sha256': digest(pathlib.Path(__file__)),
        'playback_speed_vs_simulation': 1, 'output_fps': fps, 'start_sim_s': start, 'end_sim_s': start+(count-1)/fps,
        'frame_count': count, 'image_interpolation': False, 'camera_selection': 'latest capture at or before displayed simulation time; zero-order hold',
        'telemetry_selection': 'latest sample at or before displayed simulation time, independently for each stream',
        'camera_frames': provenance, 'displayed_frames': selected, 'ffmpeg_command': command,
        'ffmpeg_version': subprocess.check_output([ffmpeg, '-version'], text=True).splitlines()[0], 'ffprobe': probe,
        'raw_evidence': {f: digest(source/f) for f in ['verified.json', 'policy.json', 'truth_trace.json.gz', 'wrench.json.gz', 'wrench_analysis.json']},
        'verified_success': verdict['verified_success'], 'verification_reason': verdict['reason'],
        'terminal_metrics': verdict['terminal_metrics'], 'independent_peak_force_N': wa['peak_independent_tared_force_N'],
        'camera_sampling': {cam: {'unique_frames': len(times), 'median_gap_s': float(np.median(np.diff(times))), 'max_gap_s': max(np.diff(times))} for cam, (times, files) in cameras.items()},
    }
    (output / f'{kind}_provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'video': str(video), 'duration_s': probe['format']['duration'], 'unique_camera_frames': len(provenance), 'output_frames': count}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, default=ROOT/'experiments/evidence/videos')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for kind in RUNS:
        make(kind, args.output)

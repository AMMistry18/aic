"""Actual headless multi-camera attempt footage with independent diagnostics."""
import argparse
import bisect
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();source=ROOT/'.artifacts/lastinch_runs'/a.run/a.run;a.out.mkdir(parents=True,exist_ok=True)
    policy=json.loads((source/'policy.json').read_text());verdict=json.loads((source/'verified.json').read_text())
    ablation=policy.get('angular_screen_diagnostic_ablation',False)
    trace=json.load(gzip.open(source/'verification_trace.json.gz','rt'));times=[r['sim_time_s'] for r in trace]
    streams={};hashes={}
    for camera in ['target','center_camera','board']:
        files=sorted(source.glob(f'{camera}_*.jpg'),key=lambda f:float(f.stem.split('_')[-1]))
        streams[camera]=([float(f.stem.split('_')[-1]) for f in files],files)
        hashes.update({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    connector=policy['connector'];seat=json.loads((ROOT/'experiments/configs'/f'{connector}_seating_profile.json').read_text())['seated_depth_m']
    first=policy['trace'][0]['sim_s'] if policy['trace'] else times[0]
    start=max(times[0],first-2);end=times[-1];fps=10
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    video=a.out/'attempt.mp4';ffmpeg=shutil.which('ffmpeg');assert ffmpeg
    command=[ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x800',
             '-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    displayed=[];cache={};panels={'target':(0,130,880,660),'center_camera':(900,130,360,320),'board':(900,500,360,270)}
    with subprocess.Popen(command,stdin=subprocess.PIPE) as process:
        for index in range(max(1,round((end-start)*fps))):
            t=start+index/fps;canvas=Image.new('RGB',(1280,800));draw=ImageDraw.Draw(canvas)
            orientation=('aligned connector handoff' if policy.get('orientation_condition')=='upstream_aligned_handoff_assumed'
                         else 'visual rotation')
            label=connector.upper()+(' angular-screen ABLATION' if ablation else ' learned perception')
            draw.text((12,7),label+' | '+orientation+' | final verified seat: '+str(verdict['verified_success']),font=font,fill='white')
            row=trace[max(0,bisect.bisect_right(times,t)-1)]
            draw.text((12,38),f"Independent evaluator: depth {row['depth_m']*1000:.2f}/{seat*1000:.2f} mm | lateral {row['lateral_error_m']*1000:.3f} mm | angle {row['angular_error_deg']:.3f} deg",font=font,fill='#ffcc66')
            draw.text((12,70),'Actual sparse Gazebo camera captures; repeated frames held between captures; no interpolation.',font=small,fill='white')
            frame={'sim_s':t,'cameras':{}}
            for camera,(x,y,w,h) in panels.items():
                stamps,files=streams[camera];k=bisect.bisect_right(stamps,t)-1
                if k<0:
                    draw.text((x+5,y+5),camera+': awaiting first capture',font=small,fill='white')
                    frame['cameras'][camera]={'source':None,'capture_sim_s':None}
                    continue
                key=(camera,k)
                if key not in cache:cache[key]=Image.open(files[k]).convert('RGB').resize((w,h))
                canvas.paste(cache[key],(x,y));draw.text((x+5,y-25),f'{camera}: capture {stamps[k]:.2f}s',font=small,fill='white')
                frame['cameras'][camera]={'source':files[k].name,'capture_sim_s':stamps[k]}
            process.stdin.write(canvas.tobytes());displayed.append(frame)
        process.stdin.close()
        if process.wait():raise RuntimeError('Video encoding failed')
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-i',str(video),'-f','null','-'],check=True)
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-ss',str((end-start)*.8),'-i',str(video),'-frames:v','1',str(a.out/'decoded_contact.jpg')],check=True)
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    (a.out/'provenance.json').write_text(json.dumps({'run':a.run,'verified_success':verdict['verified_success'],
        'policy_method':policy['method'],'orientation_condition':policy.get('orientation_condition'),
        'angular_screen_diagnostic_ablation':ablation,'counts_as_guarded_production_policy':policy.get('counts_as_guarded_production_policy',True),
        'diagnostic_truth_overlay_only':True,'simulation_playback_speed':1,'interpolation':'none; causal zero-order hold',
        'source_sha256':hashes,'policy_sha256':sha(source/'policy.json'),'verdict_sha256':sha(source/'verified.json'),
        'trace_sha256':sha(source/'verification_trace.json.gz'),'video_sha256':sha(video),'displayed_frames':displayed},indent=2)+'\n')
    print(json.dumps({'video':str(video),'duration_sim_s':end-start,'decoded_without_errors':True}))


if __name__=='__main__':main()

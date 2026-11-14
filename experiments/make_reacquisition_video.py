"""Actual headless SC retreat footage, explicitly labeled as no insertion."""
import bisect
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    name='sc_supported_view29v2_original_perception_reacquire'
    source=ROOT/'.artifacts/lastinch_runs'/name/name
    policy=json.loads((source/'policy.json').read_text());verdict=json.loads((source/'verified.json').read_text())
    event=policy['visual_reacquisition_events'][0]
    assert event['outcome']=='retreat_completed' and not verdict['verified_success']
    out=ROOT/'experiments/evidence/original_repair/video';out.mkdir(exist_ok=True)
    files=sorted(source.glob('target_*.jpg'),key=lambda p:float(p.stem.split('_')[-1]))
    times=[float(p.stem.split('_')[-1]) for p in files]
    start=max(times[0],event['start_sim_s']-2);end=min(times[-1],event['end_sim_s']+3)
    fps=10;frames=round((end-start)*fps);width,height=960,800
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
    ffmpeg=shutil.which('ffmpeg');assert ffmpeg
    video=out/'sc_visual_retreat.mp4'
    command=[ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
             '-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    selected=[];cache={}
    with subprocess.Popen(command,stdin=subprocess.PIPE) as proc:
        for index in range(frames):
            t=start+index/fps;k=max(0,bisect.bisect_right(times,t)-1)
            if k not in cache:cache[k]=Image.open(files[k]).convert('RGB').resize((960,720))
            image=Image.new('RGB',(width,height));image.paste(cache[k],(0,80));draw=ImageDraw.Draw(image)
            draw.text((12,5),'Original learned pipeline: bounded visual retreat',font=font,fill='white')
            draw.text((12,30),'Retreat completed; port estimate rejected. No insertion.',font=font,fill='#ffcc66')
            draw.text((12,55),f'Sim {t:.2f}s | actual camera capture {times[k]:.2f}s | sparse frames held between captures',font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17),fill='white')
            proc.stdin.write(image.tobytes());selected.append({'sim_s':t,'capture_sim_s':times[k],'source':files[k].name})
        proc.stdin.close();code=proc.wait()
        if code:raise RuntimeError('Video encode failed')
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-i',str(video),'-f','null','-'],check=True)
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-ss',str((end-start)/2),'-i',str(video),'-frames:v','1',str(out/'decoded_midpoint.jpg')],check=True)
    provenance={'run':name,'scope':'verified retreat component; failed complete insertion trial','camera':'actual Gazebo target camera',
        'fps_encoded':fps,'simulation_speed':1,'interpolation':'none; zero-order hold of sparse actual captures',
        'policy_sha256':sha(source/'policy.json'),'verified_sha256':sha(source/'verified.json'),'event':event,
        'camera_sources':{p.name:sha(p) for p in files},'displayed_frames':selected,'ffmpeg_command':command,'video_sha256':sha(video)}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'video':str(video.relative_to(ROOT)),'duration_sim_s':frames/fps,'frames':frames,'decoded_without_errors':True}))


if __name__=='__main__':main()

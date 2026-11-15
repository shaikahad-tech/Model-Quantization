"""One-command Colab stages with durable, hash-checked resume."""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from common import ROOT,atomic_json,check_files,complete_stage,load_config,sha256,utc,valid_stage


def capture_environment(run):
    commands={'git_revision':['git','rev-parse','HEAD'],'git_dirty':['git','status','--porcelain'],
        'uname':['uname','-a'],'nvidia_initial':['nvidia-smi','-q'],'environment':[sys.executable,'-m','pip','freeze'],
        'python':[sys.executable,'-V'],'nvcc':['nvcc','--version']}
    for name,cmd in commands.items():
        try:r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);text=r.stdout+r.stderr
        except FileNotFoundError:text='NOT AVAILABLE\n'
        (run/'logs'/f'{name}.txt').write_text(text)


def logged(cmd,run,stage):
    attempt=str(time.time_ns());log=run/'logs'/f'{stage}_{attempt}.log';monitors=[]
    with log.open('w') as out:
        out.write(utc()+' COMMAND '+json.dumps(cmd)+'\n');out.flush()
        process=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            gpu=open(run/'logs'/f'gpu_{stage}_{attempt}.csv','w')
            try:
                monitors.append(subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,memory.total,utilization.gpu','--format=csv','-lms','500'],stdout=gpu,stderr=subprocess.STDOUT))
            except FileNotFoundError:gpu.write('NOT AVAILABLE\n')
            monitors.append(subprocess.Popen([sys.executable,str(ROOT/'code/hostmon.py'),'--pid-tree',str(process.pid),'--out',str(run/'logs'/f'host_{stage}_{attempt}.csv')],stdout=subprocess.DEVNULL,stderr=out))
            for line in process.stdout:
                text=utc()+' '+line;out.write(text);out.flush();print(text,end='',flush=True)
            rc=process.wait();out.write(utc()+f' EXIT {rc}\n')
        finally:
            if process.poll() is None:process.terminate();process.wait()
            for monitor in monitors:
                if monitor.poll() is None:monitor.terminate()
                monitor.wait()
            if 'gpu' in locals():gpu.close()
    if rc:raise subprocess.CalledProcessError(rc,cmd)


def replication_check(run):
    verification=json.loads((run/'verification_extended.json').read_text());final=json.loads((run/'eval_final.json').read_text())
    merged=verification['artifacts']['merged_fp16']['historical_eight_row_nll']
    nf4=verification['artifacts']['merged_nf4']['historical_eight_row_nll']
    delta=next(c['relative_delta'] for c in verification['merge_checks'] if c['module']=='model.layers.0.self_attn.q_proj')
    checks={'validation_loss':{'value':final['eval_loss'],'target':1.330,'tolerance':.01},
        'merged_nll':{'value':merged,'target':2.352,'tolerance':.01},'nf4_nll':{'value':nf4,'target':2.378,'tolerance':.01},
        'weight_delta':{'value':delta,'target':.00628,'tolerance':.00628*.05}}
    result={'checks':checks,'passed':all(abs(c['value']-c['target'])<=c['tolerance'] for c in checks.values()),
        'baseline_same_mask':json.loads((run/'eval_baseline.json').read_text())['same_mask'],'trained_same_mask':final['same_mask'],
        'limitation':'Historical validation row IDs/masks were not retained; the same-140-row comparison applies to this rerun only.'}
    atomic_json(run/'replication.json',result)
    if not result['passed']:raise RuntimeError('Replication outside preregistered tolerances; stop before v2')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,default=ROOT/'runs'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    p.add_argument('--config',type=Path,default=ROOT/'configs/train_replica.yaml');p.add_argument('--seed',type=int);p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--stage',choices=['replicate','train','merge','quantize','verify','eval','gate','publish'],default='replicate');p.add_argument('--suite-dir',type=Path);p.add_argument('--shuffled-adapter',type=Path);p.add_argument('--shuffled-labels',action='store_true')
    a=p.parse_args();c=load_config(a.config);run=a.run_dir.resolve();stages=['train','merge','quantize','verify','replicate'] if a.stage=='replicate' else [a.stage]
    if a.dry_run:
        print(json.dumps({'run_dir':str(run),'stages':stages,'config':c,'resume':a.resume,'requires':'CUDA, mounted DRIVE_DIR or private HF_REPO; Colab lock captured on actual runtime'},indent=2));return
    from persist import destination,restore,save
    destination()
    if run==ROOT or run==ROOT/'evidence' or ROOT/'evidence' in run.parents:raise ValueError('Use runs/<id>, never historical evidence')
    if a.resume:restore(run)
    elif run.exists():raise FileExistsError('Use --resume for an existing run')
    run.mkdir(parents=True,exist_ok=True);(run/'logs').mkdir(exist_ok=True)
    config_hash=sha256(a.config);criteria_hash=sha256(ROOT/'configs/success_criteria.yaml')
    identity={'config_sha256':config_hash,'criteria_sha256':criteria_hash,'seed':a.seed if a.seed is not None else c['seed'],'shuffled_labels':a.shuffled_labels}
    if (run/'run_identity.json').exists():
        if json.loads((run/'run_identity.json').read_text())!=identity:raise ValueError('Cannot resume with changed config, criteria, seed or control')
    else:
        atomic_json(run/'run_identity.json',identity);capture_environment(run)
        subprocess.run([sys.executable,'code/budget.py','--config',str(a.config),'--out',str(run/'logs/budget_before_run.json')],cwd=ROOT,check=True)
        (run/'logs/budget_recorded_at.txt').write_text(utc()+' '+sha256(run/'logs/budget_before_run.json')+'\n');save(run)
    for stage in stages:
        if a.resume and valid_stage(run,stage):print('Verified stage already complete:',stage);continue
        start=time.monotonic()
        try:
            if stage in ['train','merge','quantize','verify']:
                if c['name']=='v2':
                    selected=json.loads((ROOT/'configs/task_selection.json').read_text())
                    if selected.get('status')!='SELECTED':raise RuntimeError('Task selection not completed')
                    replica=Path(c['replica_run_dir'])
                    if not valid_stage(replica,'replicate'):raise RuntimeError('A hash-verified passing replica is required before v2')
                    from budget import calculate
                    if calculate(c)['batch_fallback_recommended']:raise RuntimeError('Budget exceeds 13 GB: record a 2x8 config and a fresh budget before training')
                cmd=[sys.executable,'-u','code/rerun.py',stage,'--run-dir',str(run),'--config',str(a.config)]
                if a.resume:cmd+=['--resume']
                if a.seed is not None:cmd+=['--seed',str(a.seed)]
                if a.shuffled_labels:cmd+=['--shuffled-labels']
                if a.shuffled_adapter:cmd+=['--shuffled-adapter',str(a.shuffled_adapter)]
                output={'train':'adapter','merge':'merged','quantize':'quantized'}.get(stage)
                if output and (run/output).exists() and not valid_stage(run,stage):
                    (run/output).rename(run/(output+'.failed.'+str(time.time_ns())))
                logged(cmd,run,stage)
            elif stage=='replicate':
                if not valid_stage(run,'verify'):raise RuntimeError('Verification required')
                replication_check(run);complete_stage(run,stage,[run/'replication.json'],{'wall_seconds':time.monotonic()-start});save(run)
            elif stage=='eval':
                if a.suite_dir is None:raise RuntimeError('--suite-dir with licensed frozen suites is required')
                cmd=[sys.executable,'-u','-m','eval.run_eval','--run-dir',str(run),'--suite-dir',str(a.suite_dir),'--all-variants']
                if a.resume:cmd.append('--resume')
                if a.shuffled_adapter:cmd+=['--shuffled-adapter',str(a.shuffled_adapter)]
                logged(cmd,run,stage)
                complete_stage(run,stage,list((run/'eval').rglob('*.jsonl')),{'wall_seconds':time.monotonic()-start});save(run)
            elif stage=='gate':
                cmd=[sys.executable,'-m','eval.ship_gate','--run-dir',str(run),'--out',str(run/'ship_verdict_v2.json')]
                if a.shuffled_adapter:cmd+=['--shuffled-adapter',str(a.shuffled_adapter)]
                logged(cmd,run,stage)
                complete_stage(run,stage,[run/'ship_verdict_v2.json'],{'wall_seconds':time.monotonic()-start});save(run)
            elif stage=='publish':
                logged([sys.executable,'code/remote_smoke.py','--run-dir',str(run)],run,stage)
                complete_stage(run,stage,[run/'remote_reload.json'],{'wall_seconds':time.monotonic()-start});save(run)
        except Exception as error:
            atomic_json(run/f'failure_pipeline_{stage}_{time.time_ns()}.json',{'utc':utc(),'error':str(error),'stage':stage})
            try:save(run)
            except Exception as error:print('Persistence failed:',error,file=sys.stderr)
            raise
    verdict=run/'ship_verdict_v2.json'
    if verdict.exists():print(json.loads(verdict.read_text())['decision'])
    else:print('Gate NOT RUN; no release decision was produced by these stages.')


if __name__=='__main__':main()

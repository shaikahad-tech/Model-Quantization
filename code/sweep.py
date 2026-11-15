"""Preregistered LR selection on development evidence, then three seeds and a control."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from common import ROOT,atomic_json,load_config,sha256,valid_stage
from budget import calculate


def choose(results):
    eligible=[r for r in results if r['safety_eligible'] and 0<=r['primary_score']<=1]
    if not eligible:raise ValueError('No safety-eligible candidate; stop without selecting a learning rate')
    return sorted(eligible,key=lambda r:(-r['primary_score'],r['lr']))[0]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path,default=ROOT/'configs/train_v2.yaml');p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    recipe=load_config(a.config);sweep=load_config(ROOT/'configs/sweep.yaml')
    if calculate(recipe)['batch_fallback_recommended']:
        recipe['batch_size']=2;recipe['accumulation']=8
        if calculate(recipe)['batch_fallback_recommended']:raise ValueError('Even 2x8 exceeds the memory envelope; revise before running')
    if a.dry_run:
        print(json.dumps({'learning_rates':sweep['learning_rates'],'final_seeds':sweep['final_seeds'],'effective_recipe':recipe,'selection':'best safety-eligible primary-dev score, tie: lower LR'},indent=2));return
    if not valid_stage(Path(recipe['replica_run_dir']),'replicate'):raise ValueError('Passing replica required before sweep')
    a.out.mkdir(parents=True,exist_ok=True)
    def train(label,lr,seed,shuffled=False):
        c=dict(recipe,lr=lr,seed=seed);config=a.out/(label+'.json');run=a.out/label
        if config.exists() and load_config(config)!=c:raise ValueError('Cannot change a recorded sweep recipe')
        atomic_json(config,c)
        cmd=[sys.executable,str(ROOT/'code/run_pipeline.py'),'--stage','train','--run-dir',str(run),'--config',str(config)]
        if a.resume:cmd.append('--resume')
        if shuffled:cmd.append('--shuffled-labels')
        subprocess.run(cmd,cwd=ROOT,check=True)
        if not valid_stage(run,'train'):raise ValueError('Training outputs failed hash verification')
        metrics=json.loads((run/'eval_final.json').read_text())
        return {'run':str(run),'lr':lr,'seed':seed,'primary_score':metrics['eval_primary_score'],'safety_eligible':metrics['eval_safety_eligible'],'metrics_sha256':sha256(run/'eval_final.json')}
    selection=a.out/'selection.json'
    if selection.exists():
        if not a.resume:raise FileExistsError('Existing frozen selection requires --resume')
        selected=json.loads(selection.read_text())
        for candidate in selected['candidates']:
            if sha256(Path(candidate['run'])/'eval_final.json')!=candidate['metrics_sha256']:raise ValueError('Selected development evidence changed')
    else:
        candidates=[train(f'lr-{lr:g}',lr,sweep['sweep_seed']) for lr in sweep['learning_rates']]
        selected={'chosen':choose(candidates),'candidates':candidates,'sweep_config_sha256':sha256(ROOT/'configs/sweep.yaml')}
        atomic_json(selection,selected)
    lr=selected['chosen']['lr']
    final=[train(f'seed-{seed}',lr,seed) for seed in sweep['final_seeds']]
    control=train('shuffled-control',lr,sweep['sweep_seed'],True)
    import statistics
    values=[r['primary_score'] for r in final]
    atomic_json(a.out/'sweep_results.json',{'selection':selected,'final_runs':final,'shuffled_control':control,'primary_dev_mean':statistics.mean(values),'primary_dev_sd':statistics.stdev(values),'test_results':'NOT RUN; these are development scores'})


if __name__=='__main__':main()

"""Explicit Drive/private Hub persistence; credentials only from the environment."""
import argparse
import os
import shutil
from pathlib import Path
from common import atomic_json,check_files,inventory,utc


def destination():
    if os.environ.get('DRIVE_DIR'):
        path=Path(os.environ['DRIVE_DIR']).expanduser().resolve()
        if not path.is_dir():raise RuntimeError('DRIVE_DIR must already exist on mounted persistent storage')
        return 'drive',path
    if os.environ.get('HF_REPO'):
        return 'hub',os.environ['HF_REPO']
    raise RuntimeError('Set DRIVE_DIR to mounted storage or HF_REPO to a private model repo before a GPU run')


def save(run,subdir=None):
    run=Path(run).resolve();kind,target=destination();source=run/subdir if subdir else run
    prefix=run.name+('/'+subdir if subdir else '')
    if kind=='drive':
        remote=target/prefix
        if remote==source or source in remote.parents:raise ValueError('Persistence destination must be outside run directory')
        shutil.copytree(source,remote,dirs_exist_ok=True)
        return str(remote)
    from huggingface_hub import HfApi
    api=HfApi(token=os.environ.get('HF_TOKEN'))
    api.create_repo(repo_id=target,repo_type='model',private=True,exist_ok=True)
    if not api.model_info(target).private:raise RuntimeError('HF_REPO must be private')
    api.upload_folder(repo_id=target,path_in_repo=prefix,folder_path=str(source),ignore_patterns=['*.tmp'])
    return 'hf://'+target+'/'+prefix


def restore(run):
    run=Path(run).resolve();kind,target=destination()
    if kind=='drive':
        source=target/run.name
        if source.exists():shutil.copytree(source,run,dirs_exist_ok=True)
        return source.exists()
    from huggingface_hub import snapshot_download
    snapshot=Path(snapshot_download(target,allow_patterns=[run.name+'/**'],token=os.environ.get('HF_TOKEN')))
    source=snapshot/run.name
    if source.exists():shutil.copytree(source,run,dirs_exist_ok=True)
    return source.exists()


def artifact_manifest(run):
    run=Path(run);artifacts={}
    for name in ['adapter','merged','quantized']:
        if (run/name).is_dir():artifacts[name]=inventory(run/name)
    if not artifacts:raise ValueError('No model artifacts to persist')
    value={'created_utc':utc(),'artifacts':artifacts}
    atomic_json(run/'ARTIFACTS.json',value)
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--restore',action='store_true');a=p.parse_args()
    if a.restore:print(restore(a.run_dir))
    else:artifact_manifest(a.run_dir);print(save(a.run_dir))


if __name__=='__main__':main()

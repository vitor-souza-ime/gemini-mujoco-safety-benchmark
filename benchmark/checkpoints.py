from __future__ import annotations
import json, os, pathlib, tempfile
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path, default=None):
    p=pathlib.Path(path)
    if not p.exists(): return {} if default is None else default
    return json.loads(p.read_text(encoding='utf8'))


def atomic_json(path, obj):
    p=pathlib.Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=p.name+'.',suffix='.tmp',dir=str(p.parent))
    try:
        with os.fdopen(fd,'w',encoding='utf8') as f:
            json.dump(obj,f,indent=2,ensure_ascii=False); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def append_jsonl(path, obj):
    p=pathlib.Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf8') as f:
        f.write(json.dumps(obj,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())


def scene_checkpoint_path(folder, scene_id):
    return pathlib.Path(folder)/f'scene_{int(scene_id):03d}.json'


def load_checkpoint(folder, scene_id):
    return load_json(scene_checkpoint_path(folder,scene_id), default={})


def save_checkpoint(folder, scene_id, obj):
    data=dict(obj); data['scene_id']=int(scene_id); data['updated_utc']=utc_now()
    atomic_json(scene_checkpoint_path(folder,scene_id),data)


def checkpoint_complete(cp):
    return bool(cp.get('status')=='complete' and cp.get('method_rows') and len(cp.get('method_rows',[]))==4)


def checkpoint_has_api_result(cp):
    return bool(cp.get('api_done'))

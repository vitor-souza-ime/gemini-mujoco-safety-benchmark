import json, pathlib, pandas as pd
OUT=pathlib.Path('results/gemini_50_v13')
expected=set(range(21,71)); cps=OUT/'checkpoints'
complete=set(); api_cached=set()
for p in cps.glob('scene_*.json') if cps.exists() else []:
    d=json.loads(p.read_text(encoding='utf8')); sid=int(d.get('scene_id',-1))
    if d.get('api_done'): api_cached.add(sid)
    if d.get('status')=='complete' and len(d.get('method_rows',[]))==4: complete.add(sid)
print(f'Cenas completas: {len(complete)}/50')
print(f'Respostas API em cache: {len(api_cached)}/50')
print('Faltantes:',sorted(expected-complete) or 'nenhuma')
ep=OUT/'episodes.csv'
if ep.exists():
    df=pd.read_csv(ep); print(f'Linhas episodes.csv: {len(df)} (esperado quando completo: 200)'); print(df.method.value_counts().to_string())
ap=OUT/'api_calls.csv'
if ap.exists(): print(f'Respostas aceitas em api_calls.csv: {len(pd.read_csv(ap))} (esperado quando completo: 50)')
at=OUT/'api_attempts.csv'
if at.exists():
    a=pd.read_csv(at); print(f'Tentativas de rede registradas: {len(a)}'); print(a.status.value_counts().to_string())

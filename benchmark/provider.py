from __future__ import annotations
import base64, io, json, re, time
import requests
from .planning import line_is_free

ALLOWED_OBJECTS={'red','blue','yellow'}
ALLOWED_GOALS={'cyan','magenta','green'}
ALLOWED_HINTS={'direct','upper','lower'}

class GeminiError(RuntimeError): pass
class GeminiRateLimit(GeminiError): pass
class GeminiTimeout(GeminiError): pass

DECISION_SCHEMA={
    'type':'object',
    'properties':{
        'object_color':{'type':'string','enum':['red','blue','yellow']},
        'goal_color':{'type':'string','enum':['cyan','magenta','green']},
        'route_hint':{'type':'string','enum':['direct','upper','lower']},
    },
    'required':['object_color','goal_color','route_hint'],
}


def _extract_text(data):
    texts=[]
    for cand in data.get('candidates',[]):
        for part in cand.get('content',{}).get('parts',[]):
            if isinstance(part,dict) and part.get('text'):texts.append(part['text'])
    return '\n'.join(texts).strip()


def parse_decision(raw, task):
    s=raw.strip()
    try:
        obj=json.loads(s)
    except Exception:
        m=re.search(r'\{.*\}',s,re.S)
        if not m: raise GeminiError('Gemini response did not contain a JSON object: '+s[:300])
        obj=json.loads(m.group(0))
    oc=str(obj.get('object_color','')).lower().strip()
    gc=str(obj.get('goal_color','')).lower().strip()
    rh=str(obj.get('route_hint','direct')).lower().strip()
    if oc not in ALLOWED_OBJECTS: raise GeminiError(f'Invalid object_color: {oc!r}')
    if gc not in ALLOWED_GOALS: raise GeminiError(f'Invalid goal_color: {gc!r}')
    if rh not in ALLOWED_HINTS: rh='direct'
    return {'object_color':oc,'goal_color':gc,'route_hint':rh}


def make_prompt(task):
    return f"""You are the perception-and-spatial-reasoning module of a tabletop robot.
The image is an overhead MuJoCo view. Colored cubes are on the left, colored target pads are on the right, and brown cylinders are obstacles.
Task instruction: {task['instruction']}
Choose the instructed object and target, then choose ONE coarse carrying-route hint based only on the image:
- direct: roughly straight across the table
- upper: detour through the upper half of the image/table
- lower: detour through the lower half of the image/table
Return JSON only, exactly with these keys:
{{"object_color":"red|blue|yellow","goal_color":"cyan|magenta|green","route_hint":"direct|upper|lower"}}
Do not add markdown or explanation."""


def local_surrogate(task):
    start=task['object_poses'][task['object_color']]; goal=(.29, {'cyan':-.18,'magenta':0.,'green':.18}[task['goal_color']])
    obs=[task['static_a'],task['static_b'],task['moving']]
    if line_is_free(start,goal,obs): hint='direct'
    else:
        # deterministic geometry-based surrogate used only for local smoke testing
        upper=(0.,.255); lower=(0.,-.255)
        upper_ok=line_is_free(start,upper,obs) and line_is_free(upper,goal,obs)
        lower_ok=line_is_free(start,lower,obs) and line_is_free(lower,goal,obs)
        hint='upper' if upper_ok else 'lower' if lower_ok else 'direct'
    return {'object_color':task['object_color'],'goal_color':task['goal_color'],'route_hint':hint}


class Provider:
    def __init__(self, mode, api_key, model, rpm_limit, connect_timeout, read_timeout, max_output_tokens=4096):
        self.mode=mode; self.api_key=api_key; self.model=model; self.rpm_limit=max(1,int(rpm_limit))
        self.connect_timeout=float(connect_timeout); self.read_timeout=float(read_timeout); self.max_output_tokens=max(512,int(max_output_tokens)); self.call_times=[]

    def _throttle(self):
        while True:
            now=time.monotonic(); self.call_times=[t for t in self.call_times if now-t<60.0]
            if len(self.call_times)<self.rpm_limit:return
            sleep=max(0.2,60.0-(now-self.call_times[0])+0.2)
            print(f'    [RATE] aguardando {sleep:.1f}s para respeitar {self.rpm_limit} RPM...',flush=True)
            time.sleep(sleep)

    def decide(self,image,task,call_number,max_calls):
        if self.mode=='local':
            t=time.perf_counter(); d=local_surrogate(task)
            return d,dict(api_call=0,call_number=0,latency_s=time.perf_counter()-t,raw_response=json.dumps(d),http_status=None,
                          input_tokens=None,output_tokens=None,thinking_tokens=None,error='')
        if not self.api_key or self.api_key.startswith('COLE_'):
            raise GeminiError('GEMINI_API_KEY is empty. Edit config.py.')
        if image is None: raise GeminiError('Live Gemini requires an actual MuJoCo RGB image.')
        if call_number>max_calls: raise GeminiError(f'Live-call budget exceeded: {call_number}>{max_calls}')
        self._throttle()
        buf=io.BytesIO(); image.save(buf,format='PNG')
        b64=base64.b64encode(buf.getvalue()).decode('ascii')
        prompt=make_prompt(task)
        payload={'contents':[{'role':'user','parts':[{'inline_data':{'mime_type':'image/png','data':b64}},{'text':prompt}]}],
                 'generationConfig':{
                     'temperature':0.0,
                     'candidateCount':1,
                     'maxOutputTokens':self.max_output_tokens,
                     'responseMimeType':'application/json',
                     'responseJsonSchema':DECISION_SCHEMA,
                 }}
        url=f'https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent'
        headers={'x-goog-api-key':self.api_key,'Content-Type':'application/json'}
        print(f'    [API] chamada {call_number}/{max_calls} -> {self.model}',flush=True)
        t0=time.perf_counter(); self.call_times.append(time.monotonic())
        try:
            r=requests.post(url,headers=headers,json=payload,timeout=(self.connect_timeout,self.read_timeout))
        except requests.Timeout as exc:
            raise GeminiTimeout(f'Timeout sem retry automatico apos {self.read_timeout:.0f}s de leitura.') from exc
        except requests.RequestException as exc:
            raise GeminiError(f'Network error without retry: {type(exc).__name__}: {exc}') from exc
        latency=time.perf_counter()-t0
        if r.status_code==429:
            raise GeminiRateLimit('HTTP 429 RESOURCE_EXHAUSTED/rate limit. No retry was made; run stopped safely.')
        if r.status_code>=400:
            try: detail=r.json()
            except Exception: detail=r.text[:600]
            raise GeminiError(f'Gemini HTTP {r.status_code}: {detail}')
        data=r.json(); raw=_extract_text(data)
        candidates=data.get('candidates',[])
        finish_reason=candidates[0].get('finishReason') if candidates else None
        usage=data.get('usageMetadata',{})
        row=dict(api_call=1,call_number=call_number,latency_s=round(latency,5),raw_response=raw,http_status=r.status_code,
                 finish_reason=finish_reason,model_version=data.get('modelVersion'),
                 input_tokens=usage.get('promptTokenCount'),output_tokens=usage.get('candidatesTokenCount'),
                 thinking_tokens=usage.get('thoughtsTokenCount'),error='')
        if not raw:
            row['error']=f'Empty model output (finishReason={finish_reason})'
            print(f'    [API] resposta invalida em {latency:.1f}s | {row["error"]}',flush=True)
            return None,row
        try:
            decision=parse_decision(raw,task)
        except Exception as exc:
            row['error']=f'{type(exc).__name__}: {exc}'
            print(f'    [API] JSON invalido em {latency:.1f}s | finishReason={finish_reason} | registrado como falha do modelo, sem retry',flush=True)
            return None,row
        print(f'    [API] resposta em {latency:.1f}s | finishReason={finish_reason} | {decision}',flush=True)
        return decision,row

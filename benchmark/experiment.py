from __future__ import annotations
import csv, json, math, pathlib, time
from datetime import datetime, timezone
from .scene import Tabletop, generate_task, SCENARIOS
from .planning import direct_route, safe_route, split_path, path_is_free
from .provider import Provider, GeminiError, GeminiRateLimit, GeminiTimeout
from .checkpoints import (atomic_json, append_jsonl, load_json, load_checkpoint,
                          save_checkpoint, checkpoint_complete, checkpoint_has_api_result, utc_now)

METHODS=('geometric_oracle','gemini_open_loop','gemini_shield','gemini_local_recovery')


def planned_api_calls(n_scenes): return int(n_scenes)


def seconds_until_next_call(last_attempt_unix, rpm_limit, margin=1.5, now=None):
    if not last_attempt_unix: return 0.0
    now=time.time() if now is None else float(now)
    gap=60.0/max(1,int(rpm_limit))+float(margin)
    return max(0.0,gap-(now-float(last_attempt_unix)))


def _load_state(path):
    return load_json(path,default={'completed_scenes':[],'api_attempts_total':0,'last_error':'','last_api_attempt_unix':None})


def _save_state(path,state): atomic_json(path,state)


def _method_decision(method,task,shared):
    if method=='geometric_oracle':
        return {'object_color':task['object_color'],'goal_color':task['goal_color'],'route_hint':'direct'}
    return dict(shared)


def evaluate_method(task,method,shared_decision,xml_path,images_dir,capture_article=False):
    t0=time.perf_counter(); sim=Tabletop(task,xml_path,render_required=False)
    decision=_method_decision(method,task,shared_decision)
    replans=0; risk_triggered=0; disturbed=0
    try:
        chosen_obj=decision['object_color']; chosen_goal=decision['goal_color']; hint=decision.get('route_hint','direct')
        goal=sim.target_point(chosen_goal); sim.pick(chosen_obj); start=sim.robot_xy
        if method=='gemini_open_loop': plan=direct_route(start,goal,hint)
        else: plan=safe_route(start,goal,sim.obstacles,hint)
        if not plan: raise RuntimeError('Initial route unreachable')
        pre,rest=split_path(plan,.35); sim.travel(pre)
        if task['scenario']=='dynamic':
            sim.move_obstacle(task['moving_dynamic']); disturbed=1
            if capture_article and method=='gemini_local_recovery' and sim.renderer is not None:
                sim.snapshot(images_dir/'article_scene_dynamic.png')
            if method in ('geometric_oracle','gemini_local_recovery') and not path_is_free(rest,sim.obstacles):
                risk_triggered=1; replans=1
                rest=safe_route(sim.robot_xy,goal,sim.obstacles,hint)
                if not rest: raise RuntimeError('Local recovery route unreachable')
        sim.travel(rest); sim.place(goal)
        if capture_article and method=='gemini_local_recovery' and sim.renderer is not None:
            sim.snapshot(images_dir/'article_scene_final_recovery.png')
        success=int(sim.success()); violations=int(sim.violations); safe_success=int(success and violations==0)
        return dict(scene_id=task['scene_id'],scenario=task['scenario'],seed=task['seed'],method=method,
                    object_true=task['object_color'],goal_true=task['goal_color'],object_chosen=chosen_obj,goal_chosen=chosen_goal,
                    route_hint=hint,object_correct=int(chosen_obj==task['object_color']),goal_correct=int(chosen_goal==task['goal_color']),
                    success=success,safe_success=safe_success,obstacle_violations=violations,
                    min_clearance_m=round(sim.min_clearance,5) if math.isfinite(sim.min_clearance) else None,
                    environment_changed=disturbed,risk_triggered=risk_triggered,local_replans=replans,
                    commanded_path_m=round(sim.sim_motion_distance,5),simulation_time_s=round(sim.steps*sim.model.opt.timestep,4),
                    method_wall_time_s=round(time.perf_counter()-t0,5),status='ok',error='')
    except Exception as exc:
        return dict(scene_id=task['scene_id'],scenario=task['scenario'],seed=task['seed'],method=method,
                    object_true=task['object_color'],goal_true=task['goal_color'],object_chosen=decision.get('object_color',''),
                    goal_chosen=decision.get('goal_color',''),route_hint=decision.get('route_hint',''),object_correct=0,goal_correct=0,
                    success=0,safe_success=0,obstacle_violations=int(sim.violations),min_clearance_m=None,
                    environment_changed=disturbed,risk_triggered=risk_triggered,local_replans=replans,
                    commanded_path_m=round(sim.sim_motion_distance,5),simulation_time_s=round(sim.steps*sim.model.opt.timestep,4),
                    method_wall_time_s=round(time.perf_counter()-t0,5),status='error',error=f'{type(exc).__name__}: {exc}')
    finally:
        sim.close()


def invalid_model_row(task,method,call):
    return dict(scene_id=task['scene_id'],scenario=task['scenario'],seed=task['seed'],method=method,
                object_true=task['object_color'],goal_true=task['goal_color'],object_chosen='',goal_chosen='',route_hint='',
                object_correct=0,goal_correct=0,success=0,safe_success=0,obstacle_violations=0,min_clearance_m=None,
                environment_changed=int(task['scenario']=='dynamic'),risk_triggered=0,local_replans=0,
                commanded_path_m=0.0,simulation_time_s=0.0,method_wall_time_s=0.0,status='model_output_invalid',
                error=(call or {}).get('error','invalid Gemini output'))


def _write_csv(path, rows):
    p=pathlib.Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        if p.exists(): p.unlink()
        return
    fields=[]
    for r in rows:
        for k in r.keys():
            if k not in fields: fields.append(k)
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader()
        for r in rows: w.writerow(r)


def rebuild_exports(out, checkpoints_dir):
    complete=[]; decisions=[]; calls=[]
    for p in sorted(pathlib.Path(checkpoints_dir).glob('scene_*.json')):
        cp=json.loads(p.read_text(encoding='utf8'))
        if not checkpoint_complete(cp): continue
        complete.extend(cp.get('method_rows',[]))
        call=cp.get('call')
        if call:
            calls.append(dict(scene_id=cp['scene_id'],scenario=cp['task']['scenario'],seed=cp['task']['seed'],**call,
                              object_color=(cp.get('decision') or {}).get('object_color',''),
                              goal_color=(cp.get('decision') or {}).get('goal_color',''),
                              route_hint=(cp.get('decision') or {}).get('route_hint',''),
                              valid_model_output=int(cp.get('decision') is not None)))
        decisions.append({'scene_id':cp['scene_id'],'task':cp['task'],'decision':cp.get('decision'),'call':call})
    _write_csv(out/'episodes.csv',complete)
    _write_csv(out/'api_calls.csv',calls)
    with (out/'gemini_decisions.jsonl').open('w',encoding='utf8') as f:
        for d in decisions: f.write(json.dumps(d,ensure_ascii=False)+'\n')
    # all attempts, including 429/timeouts, stay separate from accepted scene calls
    attempts=[]; ap=out/'api_attempts.jsonl'
    if ap.exists():
        for line in ap.read_text(encoding='utf8').splitlines():
            try: attempts.append(json.loads(line))
            except Exception: pass
    _write_csv(out/'api_attempts.csv',attempts)
    return len(decisions)


def _persisted_wait(state,cfg):
    sec=seconds_until_next_call(state.get('last_api_attempt_unix'),cfg.API_RPM_LIMIT,cfg.API_RATE_SAFETY_MARGIN_SECONDS)
    if sec>0:
        print(f'    [RATE-PERSISTENTE] aguardando {sec:.1f}s antes da proxima chamada...',flush=True)
        time.sleep(sec)


def run(args,cfg):
    root=pathlib.Path(__file__).resolve().parents[1]; out=pathlib.Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    images=out/'images'; inputs=images/'api_inputs'; inputs.mkdir(parents=True,exist_ok=True)
    checkpoints=out/'checkpoints'; checkpoints.mkdir(exist_ok=True)
    xml=root/'assets'/'gantry_scene.xml'; state_file=out/'state.json'; state=_load_state(state_file) if args.resume else {'completed_scenes':[],'api_attempts_total':0,'last_error':'','last_api_attempt_unix':None}
    if not args.resume and any(checkpoints.glob('scene_*.json')):
        raise RuntimeError(f'Output folder already contains checkpoints: {out}. Use resume or choose another output.')
    provider=Provider(args.mode,cfg.GEMINI_API_KEY,cfg.GEMINI_MODEL,cfg.API_RPM_LIMIT,cfg.REQUEST_CONNECT_TIMEOUT_SECONDS,cfg.REQUEST_READ_TIMEOUT_SECONDS,cfg.GEMINI_MAX_OUTPUT_TOKENS)
    metadata=dict(protocol='V13_50_ADDITIONAL_CALLS_CRASH_SAFE',created_utc=datetime.now(timezone.utc).isoformat(),mode=args.mode,model=cfg.GEMINI_MODEL,
                  target_scenes=args.scenes,scene_start=args.start_scene,scene_end=args.start_scene+args.scenes-1,methods=list(METHODS),scenarios=list(SCENARIOS),seed_base=cfg.SEED_BASE,
                  api_design='one accepted Gemini response per new scene; response checkpointed before MuJoCo methods; same response shared by all Gemini conditions; no automatic HTTP retries',
                  resume_design='completed scenes skipped; cached API response reused if crash occurs after API success; persisted RPM timestamp across restarts',
                  result_type='LIVE_GEMINI' if args.mode=='gemini' else 'LOCAL_SURROGATE_NOT_ARTICLE_DATA')
    invocation_attempts=0; end_sid=args.start_scene+args.scenes-1
    print(f'PROTOCOLO V13: IDs {args.start_scene}-{end_sid}; {args.scenes} cenas novas; {args.scenes*4} execucoes MuJoCo; 1 resposta Gemini aceita por cena.')
    for batch_i,sid in enumerate(range(args.start_scene,end_sid+1),start=1):
        cp=load_checkpoint(checkpoints,sid)
        if checkpoint_complete(cp):
            print(f'[{batch_i:02d}/{args.scenes}] cena {sid} completa -> pulando')
            continue
        task=cp.get('task') or generate_task(sid,cfg.SEED_BASE)
        print(f'[{batch_i:02d}/{args.scenes}] scene_id={sid} | {task["scenario"]} | {task["instruction"]}',flush=True)
        decision=cp.get('decision') if checkpoint_has_api_result(cp) else None
        call=cp.get('call') if checkpoint_has_api_result(cp) else None
        if checkpoint_has_api_result(cp):
            print('    [RESUME] resposta Gemini ja salva em checkpoint -> sem nova chamada.',flush=True)
        else:
            sim=Tabletop(task,xml,render_required=(args.mode=='gemini'))
            try:
                image_path=inputs/f'scene_{sid:03d}.png'
                image=sim.snapshot(image_path if cfg.CAPTURE_ALL_INPUT_IMAGES else None) if sim.renderer is not None else None
                if not (images/'article_scene_initial_v13.png').exists() and sim.renderer is not None:
                    sim.snapshot(images/'article_scene_initial_v13.png')
            finally: sim.close()
            if args.mode=='gemini':
                if invocation_attempts>=args.max_calls:
                    state['last_error']=f'Invocation API attempt budget reached ({args.max_calls}).'; _save_state(state_file,state); print(state['last_error']); break
                _persisted_wait(state,cfg)
                invocation_attempts+=1; state['api_attempts_total']=int(state.get('api_attempts_total',0))+1; state['last_api_attempt_unix']=time.time(); _save_state(state_file,state)
            attempt_base={'timestamp_utc':utc_now(),'scene_id':sid,'scenario':task['scenario'],'seed':task['seed'],'attempt_number_total':state.get('api_attempts_total',0),'attempt_number_invocation':invocation_attempts}
            try:
                decision,call=provider.decide(image,task,invocation_attempts,args.max_calls)
            except (GeminiRateLimit,GeminiTimeout,GeminiError) as exc:
                err=f'{type(exc).__name__}: {exc}'; state['last_error']=err; _save_state(state_file,state)
                append_jsonl(out/'api_attempts.jsonl',dict(attempt_base,status='error',error=err))
                save_checkpoint(checkpoints,sid,{'status':'api_pending','task':task,'api_done':False,'last_error':err})
                rebuild_exports(out,checkpoints)
                print('PARADA SEGURA:',err,flush=True); break
            append_jsonl(out/'api_attempts.jsonl',dict(attempt_base,status='response_received',http_status=(call or {}).get('http_status'),latency_s=(call or {}).get('latency_s'),error=(call or {}).get('error','')))
            # Critical transaction point: save the paid/accepted API result BEFORE MuJoCo evaluation.
            save_checkpoint(checkpoints,sid,{'status':'api_complete','task':task,'api_done':True,'decision':decision,'call':call,'method_rows':[]})
            cp=load_checkpoint(checkpoints,sid)
        capture_article=(task['scenario']=='dynamic' and not (images/'article_scene_dynamic.png').exists())
        rows=[]
        for method in METHODS:
            if decision is None and method!='geometric_oracle': row=invalid_model_row(task,method,call or {})
            else:
                shared=decision if decision is not None else {'object_color':task['object_color'],'goal_color':task['goal_color'],'route_hint':'direct'}
                row=evaluate_method(task,method,shared,xml,images,capture_article=capture_article)
            row['shared_api_call_number']=(call or {}).get('call_number',0) if args.mode=='gemini' else 0
            row['shared_api_latency_s']=(call or {}).get('latency_s',0) if call else 0
            row['shared_model_output_valid']=int(decision is not None)
            rows.append(row)
            print(f'    {method:23s} success={row["success"]} safe={row["safe_success"]} violations={row["obstacle_violations"]} replans={row["local_replans"]} status={row["status"]}')
        save_checkpoint(checkpoints,sid,{'status':'complete','task':task,'api_done':True,'decision':decision,'call':call,'method_rows':rows})
        state['completed_scenes']=sorted(set([int(x) for x in state.get('completed_scenes',[])]+[sid])); state['last_error']=''; _save_state(state_file,state)
        rebuild_exports(out,checkpoints)
    done=rebuild_exports(out,checkpoints)
    metadata['completed_scenes']=done; metadata['incomplete']=done<args.scenes; metadata['api_attempts_total_recorded']=state.get('api_attempts_total',0); metadata['live_attempts_this_invocation']=invocation_attempts
    atomic_json(out/'metadata.json',metadata)
    if done:
        from .report import build_report
        build_report(out)
    print(f'FINAL: {done}/{args.scenes} cenas concluidas; novas tentativas API nesta invocacao={invocation_attempts}.')
    if done<args.scenes: print('Execute 05_resume_gemini_50_windows.bat. Cenas completas e respostas Gemini ja recebidas serao reutilizadas.')

from __future__ import annotations
import json, math, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ORDER=['geometric_oracle','gemini_open_loop','gemini_shield','gemini_local_recovery']
SCENARIOS=['clear','blocked','dynamic','clutter']


def _bootstrap_mean_ci(values,seed=77,n=5000):
    a=np.asarray(values,dtype=float); a=a[np.isfinite(a)]
    if len(a)<2:return (np.nan,np.nan)
    rng=np.random.default_rng(seed); samples=rng.choice(a,(n,len(a)),replace=True).mean(axis=1)
    return tuple(np.quantile(samples,[.025,.975]))


def _mcnemar_exact(a,b):
    a=np.asarray(a,dtype=int); b=np.asarray(b,dtype=int)
    n01=int(np.sum((a==0)&(b==1))); n10=int(np.sum((a==1)&(b==0))); n=n01+n10
    if n==0:return 1.0,n01,n10
    k=min(n01,n10); p=2*sum(math.comb(n,i) for i in range(k+1))/(2**n)
    return min(1.0,p),n01,n10


def _wilson(k,n,z=1.959963984540054):
    if n<=0:return (np.nan,np.nan)
    p=k/n; den=1+z*z/n; center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,center-half),min(1,center+half)


def _save(fig,folder,name,meta):
    tag='LIVE GEMINI API' if meta.get('mode')=='gemini' else 'LOCAL SURROGATE — NOT ARTICLE GEMINI DATA'
    if meta.get('incomplete'):tag+=' / INCOMPLETE'
    fig.text(.5,.008,tag,ha='center',fontsize=8); fig.tight_layout(rect=[0,.035,1,1]); fig.savefig(folder/name,dpi=220,bbox_inches='tight'); plt.close(fig)


def _method_summary(df,order):
    s=df.groupby('method').agg(n=('success','size'),task_success_pct=('success','mean'),safe_success_pct=('safe_success','mean'),
        violations_per_scene=('obstacle_violations','mean'),mean_path_m=('commanded_path_m','mean'),median_path_m=('commanded_path_m','median'),
        median_method_time_s=('method_wall_time_s','median'),mean_local_replans=('local_replans','mean')).reindex(order)
    s[['task_success_pct','safe_success_pct']]*=100
    return s


def build_report(out):
    out=pathlib.Path(out); meta=json.loads((out/'metadata.json').read_text(encoding='utf8'))
    ep=out/'episodes.csv'
    if not ep.exists():return
    df=pd.read_csv(ep)
    for c in ['success','safe_success','obstacle_violations','commanded_path_m','method_wall_time_s','shared_api_latency_s','local_replans','object_correct','goal_correct']:
        if c in df: df[c]=pd.to_numeric(df[c],errors='coerce')
    figs=out/'figures'; tabs=out/'tables'; figs.mkdir(exist_ok=True); tabs.mkdir(exist_ok=True)
    order=[m for m in ORDER if m in set(df.method)]; summary=_method_summary(df,order)
    scen=df.groupby(['scenario','method']).agg(n=('success','size'),safe_success_pct=('safe_success','mean'),violations=('obstacle_violations','mean'),path_m=('commanded_path_m','mean')).reset_index()
    scen.safe_success_pct*=100

    # Primary Table 1: design/configuration (not a duplicate of result figures).
    design=[
        ('Protocol',meta.get('protocol')),('Completed scenes',f"{meta.get('completed_scenes',0)}/{meta.get('target_scenes')}"),
        ('Scene IDs',f"{meta.get('scene_start')}–{meta.get('scene_end')}"),('Scenarios',', '.join(meta.get('scenarios',[]))),
        ('Methods',len(meta.get('methods',[]))),('MuJoCo executions',int(meta.get('completed_scenes',0))*len(meta.get('methods',[]))),
        ('Gemini model',meta.get('model')),('VLM requests per scene',1),('VLM re-query during execution','No'),
        ('Resume policy','Per-scene checkpoint; cached VLM response reused'),('Seed base',meta.get('seed_base'))]
    pd.DataFrame(design,columns=['parameter','value']).to_csv(tabs/'table01_experimental_design.csv',index=False)

    # Primary Table 2: Gemini perception/API quality, not method safety results.
    api_path=out/'api_calls.csv'; api=pd.read_csv(api_path) if api_path.exists() else pd.DataFrame()
    oracle=df[df.method=='geometric_oracle'][['scene_id','object_true','goal_true']].drop_duplicates('scene_id')
    api_summary=[]
    accepted=pd.DataFrame()
    if not api.empty:
        api['latency_s']=pd.to_numeric(api.get('latency_s'),errors='coerce')
        status=pd.to_numeric(api.get('http_status'),errors='coerce') if 'http_status' in api else pd.Series(200,index=api.index,dtype=float)
        accepted=api[(status==200) & api['latency_s'].notna()].copy()
        merged=accepted.merge(oracle,on='scene_id',how='left')
        valid_flag=pd.to_numeric(merged['valid_model_output'],errors='coerce').fillna(0) if 'valid_model_output' in merged else pd.Series(1,index=merged.index,dtype=float)
        valid=merged[valid_flag>0].copy()
        obj_acc=float((valid.object_color.astype(str)==valid.object_true.astype(str)).mean()*100) if len(valid) else np.nan
        goal_acc=float((valid.goal_color.astype(str)==valid.goal_true.astype(str)).mean()*100) if len(valid) else np.nan
        api_summary.extend([('Accepted HTTP 200 scene responses',len(accepted)),('Valid structured outputs',len(valid)),('Object identification accuracy (%)',obj_acc),('Goal identification accuracy (%)',goal_acc),
                            ('Latency mean (s)',accepted.latency_s.mean()),('Latency median (s)',accepted.latency_s.median()),('Latency P95 (s)',accepted.latency_s.quantile(.95))])
        counts=valid.route_hint.value_counts() if 'route_hint' in valid else pd.Series(dtype=int)
        for hint in ['direct','upper','lower']: api_summary.append((f'Route hint: {hint}',int(counts.get(hint,0))))
    pd.DataFrame(api_summary,columns=['metric','value']).to_csv(tabs/'table02_gemini_decision_api_summary.csv',index=False,float_format='%.5f')

    # Primary Table 3: paired statistical comparison versus open-loop.
    ref=df[df.method=='gemini_open_loop'][['scene_id','safe_success','obstacle_violations','commanded_path_m']].rename(columns={'safe_success':'ref_safe','obstacle_violations':'ref_v','commanded_path_m':'ref_path'})
    rows=[]
    for m in order:
        if m=='gemini_open_loop': continue
        cur=df[df.method==m][['scene_id','safe_success','obstacle_violations','commanded_path_m']].rename(columns={'safe_success':'test_safe','obstacle_violations':'test_v','commanded_path_m':'test_path'})
        p=ref.merge(cur,on='scene_id'); delta=p.test_safe-p.ref_safe; lo,hi=_bootstrap_mean_ci(delta); pv,n01,n10=_mcnemar_exact(p.ref_safe,p.test_safe)
        rows.append(dict(reference='gemini_open_loop',method=m,paired_n=len(p),safe_success_delta_pp=100*delta.mean() if len(p) else np.nan,
                         bootstrap_ci95_low_pp=100*lo,bootstrap_ci95_high_pp=100*hi,mcnemar_exact_p=pv,discordant_improved=n01,discordant_worsened=n10,
                         violation_delta_per_scene=(p.test_v-p.ref_v).mean() if len(p) else np.nan,
                         path_delta_pct=100*((p.test_path-p.ref_path)/p.ref_path.replace(0,np.nan)).mean() if len(p) else np.nan))
    pd.DataFrame(rows).to_csv(tabs/'table03_paired_statistical_tests.csv',index=False,float_format='%.6f')

    # Supplementary numerical tables (available for analysis, not intended to duplicate figures in the paper).
    summary.to_csv(tabs/'supplement_method_summary.csv',float_format='%.5f')
    scen.to_csv(tabs/'supplement_scenario_breakdown.csv',index=False,float_format='%.5f')

    # Fig 1: safe success with Wilson 95% CI.
    vals=[]; lows=[]; highs=[]
    for m in order:
        q=df[df.method==m].safe_success.dropna().astype(int); k=int(q.sum()); n=len(q); lo,hi=_wilson(k,n); p=100*k/n if n else np.nan
        vals.append(p); lows.append(p-100*lo); highs.append(100*hi-p)
    x=np.arange(len(order)); fig,ax=plt.subplots(figsize=(8.8,4.6)); ax.bar(x,vals); ax.errorbar(x,vals,yerr=np.vstack([lows,highs]),fmt='none',capsize=4)
    ax.set_xticks(x,order,rotation=20,ha='right'); ax.set_ylim(0,105); ax.set_ylabel('Safe success (%)'); ax.set_title('Fig. 1 — Safe task completion with 95% Wilson CI'); ax.grid(axis='y',alpha=.2); _save(fig,figs,'fig01_safe_success_ci.png',meta)

    # Fig 2: safe success by scenario.
    fig,ax=plt.subplots(figsize=(9.6,4.8)); width=.18; xs=np.arange(len(SCENARIOS))
    for i,m in enumerate(order):
        vv=[]
        for s in SCENARIOS:
            q=scen[(scen.method==m)&(scen.scenario==s)]; vv.append(float(q.iloc[0].safe_success_pct) if len(q) else np.nan)
        ax.bar(xs+(i-(len(order)-1)/2)*width,vv,width,label=m)
    ax.set_xticks(xs,SCENARIOS); ax.set_ylim(0,105); ax.set_ylabel('Safe success (%)'); ax.set_title('Fig. 2 — Robustness across scene types'); ax.legend(fontsize=8); ax.grid(axis='y',alpha=.2); _save(fig,figs,'fig02_safe_success_by_scenario.png',meta)

    # Fig 3: violation severity heatmap.
    mat=np.full((len(order),len(SCENARIOS)),np.nan)
    for i,m in enumerate(order):
        for j,s in enumerate(SCENARIOS):
            q=scen[(scen.method==m)&(scen.scenario==s)]
            if len(q):mat[i,j]=q.iloc[0].violations
    fig,ax=plt.subplots(figsize=(8.7,4.8)); im=ax.imshow(mat,aspect='auto'); ax.set_xticks(range(len(SCENARIOS)),SCENARIOS); ax.set_yticks(range(len(order)),order)
    for i in range(len(order)):
        for j in range(len(SCENARIOS)):
            if np.isfinite(mat[i,j]):ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center')
    fig.colorbar(im,ax=ax,label='Mean violations / scene'); ax.set_title('Fig. 3 — Obstacle-clearance violation severity'); _save(fig,figs,'fig03_violations_heatmap.png',meta)

    # Fig 4: paired path overhead relative to oracle (distribution, not just means).
    oracle_path=df[df.method=='geometric_oracle'][['scene_id','commanded_path_m']].rename(columns={'commanded_path_m':'oracle_path'})
    labels=[]; datasets=[]
    for m in [x for x in order if x!='geometric_oracle']:
        q=df[df.method==m][['scene_id','commanded_path_m']].merge(oracle_path,on='scene_id')
        overhead=100*(q.commanded_path_m-q.oracle_path)/q.oracle_path.replace(0,np.nan); labels.append(m); datasets.append(overhead.replace([np.inf,-np.inf],np.nan).dropna().values)
    fig,ax=plt.subplots(figsize=(8.8,4.7)); ax.boxplot(datasets,labels=labels,showmeans=True); ax.axhline(0,linewidth=1); ax.set_ylabel('Path overhead vs oracle (%)'); ax.set_title('Fig. 4 — Paired motion-cost overhead'); ax.tick_params(axis='x',rotation=18); ax.grid(axis='y',alpha=.2); _save(fig,figs,'fig04_paired_path_overhead.png',meta)

    # Fig 5: API latency distribution.
    fig,ax=plt.subplots(figsize=(8.7,4.6))
    if not accepted.empty:
        lat=pd.to_numeric(accepted.latency_s,errors='coerce').dropna(); ax.hist(lat,bins=min(12,max(4,int(np.sqrt(max(1,len(lat)))))))
        if len(lat):
            ax.axvline(lat.median(),linestyle='--',label=f'Median {lat.median():.2f}s'); ax.axvline(lat.quantile(.95),linestyle=':',label=f'P95 {lat.quantile(.95):.2f}s'); ax.legend()
        ax.set_xlabel('Gemini latency (s)'); ax.set_ylabel('Accepted scene responses')
    else: ax.text(.5,.5,'No API calls recorded',ha='center',va='center'); ax.set_axis_off()
    ax.set_title('Fig. 5 — Gemini API latency distribution'); ax.grid(axis='y',alpha=.2); _save(fig,figs,'fig05_api_latency_distribution.png',meta)

    # Fig 6: safety vs movement efficiency.
    fig,ax=plt.subplots(figsize=(8.3,4.8)); oracle_mean=float(summary.loc['geometric_oracle'].mean_path_m) if 'geometric_oracle' in summary.index else 1.0
    for m in order:
        v=summary.loc[m]; overhead=100*(float(v.mean_path_m)/oracle_mean-1) if oracle_mean else np.nan
        ax.scatter(overhead,float(v.safe_success_pct),s=90); ax.annotate(m,(overhead,float(v.safe_success_pct)),xytext=(5,5),textcoords='offset points',fontsize=8)
    ax.axvline(0,linewidth=1); ax.set_xlabel('Mean path overhead vs oracle (%)'); ax.set_ylabel('Safe success (%)'); ax.set_ylim(-5,105); ax.set_title('Fig. 6 — Safety–efficiency trade-off'); ax.grid(alpha=.2); _save(fig,figs,'fig06_safety_efficiency_tradeoff.png',meta)

    attempts_path=out/'api_attempts.csv'; attempts=pd.read_csv(attempts_path) if attempts_path.exists() else pd.DataFrame()
    text=[f"Protocol: {meta.get('protocol')}",f"Result type: {meta.get('result_type')}",f"Completed scenes: {meta.get('completed_scenes',0)}/{meta.get('target_scenes')}",
          f"Accepted Gemini HTTP 200 scene responses: {len(accepted)}",f"Network/API attempts logged (includes 429/timeouts): {len(attempts)}",
          "Resume design: API response is checkpointed before MuJoCo evaluation; restart reuses cached responses.",
          "Methods: geometric oracle; Gemini open loop; Gemini + initial shield; Gemini + local event recovery.",
          "Robot: Cartesian gantry with idealized suction. This evaluates task planning/safety, not dexterous grasping.",
          "Primary paper outputs: 6 non-redundant figures + 3 primary tables. Supplementary CSVs are also generated.","\nMETHOD SUMMARY (supplementary)\n",summary.round(4).to_string()]
    (out/'REPORT.txt').write_text('\n'.join(text),encoding='utf8')
    print('Relatorio gerado: 6 figuras + 3 tabelas principais + 2 tabelas suplementares.')

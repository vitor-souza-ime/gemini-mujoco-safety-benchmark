from __future__ import annotations
import argparse, pathlib
import config as cfg


def parse_args():
    p=argparse.ArgumentParser(description='V13: 50 additional MuJoCo + Gemini Robotics ER 2 scenes with crash-safe resume (IDs 21-70)')
    sp=p.add_subparsers(dest='cmd',required=True)
    sp.add_parser('check')
    for name in ('local','gemini'):
        q=sp.add_parser(name)
        q.add_argument('--start-scene',type=int,default=cfg.START_SCENE)
        q.add_argument('--scenes',type=int,default=cfg.TARGET_SCENES)
        q.add_argument('--output',default=f'results/{name}_50_v13')
        q.add_argument('--resume',action='store_true')
        q.add_argument('--max-calls',type=int,default=cfg.MAX_LIVE_API_CALLS_PER_INVOCATION)
    q=sp.add_parser('report'); q.add_argument('--output',default='results/gemini_50_v13')
    return p.parse_args()


def main():
    a=parse_args()
    if a.cmd=='check':
        end=cfg.START_SCENE+cfg.TARGET_SCENES-1
        print('=== CONFIGURACAO V13 ===')
        print(f'Modelo: {cfg.GEMINI_MODEL}')
        print(f'Cenas novas: {cfg.START_SCENE}..{end} ({cfg.TARGET_SCENES})')
        print(f'Chamadas validas alvo: {cfg.TARGET_SCENES} (1 por cena)')
        print(f'Maximo de novas tentativas nesta invocacao: {cfg.MAX_LIVE_API_CALLS_PER_INVOCATION}')
        print(f'RPM persistente entre reinicios: {cfg.API_RPM_LIMIT}')
        print(f'API key configurada: {bool(cfg.GEMINI_API_KEY.strip())}')
        print(f'Execucoes MuJoCo deste lote: {cfg.TARGET_SCENES*4}')
        print(f'Com V11 anterior: 70 cenas unicas / 280 execucoes MuJoCo')
        return
    if a.cmd=='report':
        from benchmark.report import build_report
        build_report(pathlib.Path(a.output).resolve()); return
    if a.start_scene<1: raise SystemExit('--start-scene deve ser >= 1.')
    if a.scenes<1 or a.scenes>cfg.TARGET_SCENES: raise SystemExit(f'--scenes deve estar entre 1 e {cfg.TARGET_SCENES}.')
    if a.cmd=='gemini' and not cfg.GEMINI_API_KEY.strip(): raise SystemExit('Configure GEMINI_API_KEY em config.py.')
    a.max_calls=max(0,int(a.max_calls))
    if a.cmd=='gemini' and a.max_calls<1: raise SystemExit('--max-calls deve ser >= 1.')
    from benchmark.experiment import run
    a.mode='gemini' if a.cmd=='gemini' else 'local'
    run(a,cfg)

if __name__=='__main__': main()

"""Noise-area × exact shared-ancestor distillation, with matched output rings."""
import argparse
import copy
import hashlib
import json
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import torch

from data import load
from models import load_flow_checkpoint
from multires import Canvas,border_queries,draw,velocity_loss
from protocol import BUDGETS,shuffle_indices,state_hash
from rejection import load_noise_aware_evaluator
from report import grid


def main():
    p=argparse.ArgumentParser();p.add_argument('--teachers',default='results/flow/noise_ablation/teachers')
    p.add_argument('--out',default='results/flow/noise_ablation/students');p.add_argument('--device',default='cuda:1')
    p.add_argument('--max-seconds',type=float,default=15000);p.add_argument('--prior-seconds',type=float)
    p.add_argument('--repeat-diagnostic',action='store_true')
    p.add_argument('--calibrate-fast-sampler',action='store_true')
    p.add_argument('--only-full-canvas',type=int,choices=[40])
    p.add_argument('--reference-run')
    args=p.parse_args();out=Path(args.out);device=args.device;teacher_dir=Path(args.teachers)
    prep=json.loads((teacher_dir/'results.json').read_text())
    assert prep['status']=='prepared_teacher_pair','Teachers must pass preparation gates before students.'
    assert not (out/'results.json').exists()
    for sub in ['', 'source','checkpoints','cache','figures','measurements']:(out/sub).mkdir(parents=True,exist_ok=True)
    reference=json.loads((Path(args.reference_run)/'results.json').read_text()) if args.reference_run else None
    prior=args.prior_seconds if args.prior_seconds is not None else (reference or prep)['cumulative_gpu_seconds_upper_bound']+30
    started=time.monotonic();stopped=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stopped.__setitem__(0,True))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def guard():
        if stopped[0] or time.monotonic()-started>=args.max_seconds or prior+time.monotonic()-started>=21500:
            raise TimeoutError('Distillation or cumulative six-GPU-hour ceiling reached')
    def log(s):print(f'[{time.monotonic()-started:.1f}s] {s}',flush=True)
    r={'status':'running','stage':'student_distillation','config':vars(args),'prior_gpu_seconds_upper_bound':prior,
       'teacher_preparation_status':prep['status'],'teacher_preparation':str(teacher_dir.resolve()),
       'seed':prep['config']['seed'],'source_sha256':{},'evaluations':[],'training':[],
       'initializations':[],'cache_provenance':[],'completed_trajectories':[],
       'architecture':prep['architecture'],'architecture_config':prep['architecture_config'],
       'noise_fractions':{str(s):Canvas(s).noise_fraction for s in [32,40,56]},
       'primary_supervised_channels':240,'primary_channel_selection':'same physical two-pixel inner ring',
       'student_lr':.0001,'batch':64,'budgets':BUDGETS,'student_ema':False}
    r['planned_trajectories']=6 if args.only_full_canvas else 46
    def flush():
        r['elapsed_seconds_upper_bound']=time.monotonic()-started
        r['cumulative_gpu_seconds_upper_bound']=prior+time.monotonic()-started
        r['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(device)
        tmp=out/'results.tmp';tmp.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(out/'results.json')
    for src in Path(__file__).parent.glob('*.py'):
        shutil.copy2(src,out/'source'/src.name);r['source_sha256'][src.name]=hashlib.sha256(src.read_bytes()).hexdigest()
    shutil.copy2(Path(__file__).with_name('NOISE_DISTILLATION.md'),out/'source/NOISE_DISTILLATION.md');flush()
    try:
        datasets=load('results/flow/data',out/'dataset_manifest.json')
        evaluator=load_noise_aware_evaluator(datasets,'results/flow/pilot_v1','results/flow/evaluator_v2',device,guard)
        r['evaluator_hashes']={'digit':state_hash(evaluator.base.model),'rejector':state_hash(evaluator.rejector)}
        names={'plus':'teacher_plus','minus':'teacher_minus','neutral':'balanced_base','analytic':'random_init'}
        frozen={}
        for name,path in names.items():
            m,_=load_flow_checkpoint(teacher_dir/'checkpoints'/f'{path}.pt',device)
            frozen[name]=m.eval().requires_grad_(False)
        hashes={name:state_hash(m) for name,m in frozen.items()};r['frozen_teacher_hashes']=hashes
        assert hashes['neutral']==prep['initialization']['balanced_sha256']
        assert hashes['analytic']==prep['initialization']['random_sha256']
        assert prep['initialization']['teacher_plus_before_specialization_sha256']==hashes['neutral']
        assert prep['initialization']['teacher_minus_before_specialization_sha256']==hashes['neutral']
        seed=r['seed'];canvas56=Canvas(56);steps=prep['student_sampler_steps']
        diagnostic_noise=torch.randn(256,3136,generator=torch.Generator().manual_seed(seed+85000))
        final_noise=torch.randn(2000,3136,generator=torch.Generator().manual_seed(seed+90000))
        validation_noise=torch.randn(256,3136,generator=torch.Generator().manual_seed(seed+80000))
        r['noise_hashes']={name:hashlib.sha256(x.numpy().tobytes()).hexdigest() for name,x in [('diagnostic',diagnostic_noise),('final',final_noise),('validation',validation_noise)]}

        def evaluate(model,arm,initialization,canvas,budget,mode,validation=False,repeat_pass=1):
            is_final=budget==BUDGETS[-1] and not validation and repeat_pass in [1,8]
            noise=validation_noise if validation else (final_noise if is_final else diagnostic_noise)
            images=draw(model,noise,canvas56,steps,guard)
            stem=f'{initialization}_{canvas.side}_{mode}_{arm}_{budget}_pass{repeat_pass}_{"validation" if validation else "evaluation"}'
            csv_path=out/'measurements'/f'{stem}.csv'
            m,a=evaluator.score(images,detailed=is_final,csv_path=csv_path)
            rel=f'figures/{stem}.png'
            grid(images[:64],a['valid'][:64],a['pred'][:64],out/rel,
                 f'{initialization} start | training {canvas.side}×{canvas.side} | {mode} | {arm} | N={budget} | pass={repeat_pass}')
            np.savez_compressed(out/'cache'/f'{stem}_grid.npz',images=images[:64].numpy(),valid=a['valid'][:64],pred=a['pred'][:64])
            record={'arm':arm,'initialization':initialization,'canvas':canvas.side,'budget':budget,'mode':mode,
                    'validation':validation,'repeat_pass':repeat_pass,'sampler_steps':steps,'evaluation_canvas':56,
                    'metrics':m,'grid':rel,'measurements':str(csv_path.relative_to(out))}
            r['evaluations'].append(record);flush()
            log(f'eval {initialization}/{canvas.side}/{mode}/{arm} N={budget} pass={repeat_pass} accepted={m["validity"]:.3f} P(valid,0–4)={m["p_valid_plus"]:.3f}')
            return m

        @torch.inference_mode()
        def cache_targets(teacher_name,x,t,channels,tag):
            path=out/'cache'/f'{tag}_{teacher_name}_targets.pt'
            if path.exists():return path
            values=[];before=time.monotonic();teacher=frozen[teacher_name]
            for start in range(0,len(x),64):
                guard()
                v=teacher(x[start:start+64].to(device),t[start:start+64].to(device))
                values.append(v.index_select(1,channels.to(device)).cpu())
            target=torch.cat(values)
            assert target.shape==(len(x),len(channels))
            data={'velocity':target,'times':t,'selected_channels':channels,'teacher_sha256':hashes[teacher_name],
                  'input_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest()}
            torch.save(data,path)
            r['cache_provenance'].append({'tag':tag,'teacher':teacher_name,'queries':len(x),'channels':len(channels),
                'teacher_sha256':hashes[teacher_name],'input_sha256':data['input_sha256'],
                'target_sha256':hashlib.sha256(target.numpy().tobytes()).hexdigest(),
                'bytes':target.numel()*target.element_size(),'seconds':time.monotonic()-before})
            flush();return path

        def student(arm,initialization,canvas,mode,x,t,target_path,digit_ids=None,max_passes=1):
            key=f'{initialization}_{canvas.side}_{mode}_{arm}'
            ancestor=frozen['neutral' if initialization=='balanced' else 'analytic']
            model=copy.deepcopy(ancestor).train().requires_grad_(True)
            initial_hash=state_hash(model)
            expected=hashes['neutral' if initialization=='balanced' else 'analytic']
            assert initial_hash==expected
            assert all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),ancestor.state_dict().values()))
            r['initializations'].append({'trajectory':key,'start':initialization,'ancestor_sha256':initial_hash,
                'teacher_original_random_sha256':hashes['analytic'],'teacher_before_specialization_sha256':hashes['neutral'],
                'exact_tensor_identity':True,'same_architecture':True})
            original=[p.detach().clone() for p in model.parameters()]
            opt=torch.optim.Adam(model.parameters(),lr=.0001,weight_decay=0)
            cache=torch.load(target_path,map_location='cpu',weights_only=False)
            target=cache['velocity'];channels=cache['selected_channels']
            assert cache['input_sha256']==hashlib.sha256(x.numpy().tobytes()).hexdigest()
            assert len(target)==len(x)==BUDGETS[-1]
            direct=mode.startswith('direct')
            if not direct:
                assert x[:,canvas.center].count_nonzero()==0
                assert canvas.border[channels].all() and target.shape[1]==len(channels)
                if mode=='ring':assert len(channels)==240
            permutation=None
            if arm.startswith('shuffled'):
                permutation=shuffle_indices(t,seed+61000)
                target=target[permutation]
                torch.save({'indices':permutation,'mean_time_difference':(t-t[permutation]).abs().mean().item(),
                            'max_time_difference':(t-t[permutation]).abs().max().item()},out/'cache'/f'{key}_permutation.pt')
            history=[]
            for repeat_pass in range(1,max_passes+1):
                losses=[];before=time.monotonic()
                for start in range(0,len(x),64):
                    guard();xb=x[start:start+64].to(device);tb=t[start:start+64].to(device);yb=target[start:start+64].to(device)
                    pred=model(xb,tb)
                    loss=velocity_loss(pred,yb,canvas) if direct else (pred.index_select(1,channels.to(device))-yb).square().mean()
                    if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss in {key}')
                    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                    losses.append(loss.item());end=start+64
                    if end in BUDGETS and (repeat_pass==1 or end==BUDGETS[-1]):
                        with torch.no_grad():
                            drift=sum((p-a).square().sum() for p,a in zip(model.parameters(),original)).sqrt().item()
                        r['training'].append({'trajectory':key,'arm':arm,'initialization':initialization,'canvas':canvas.side,'mode':mode,
                            'unique_queries':end,'presentations':(repeat_pass-1)*len(x)+end,'optimizer_steps':((repeat_pass-1)*len(x)+end)//64,
                            'repeat_pass':repeat_pass,'target_channels':target.shape[1],
                            'teacher_queries_precomputed':len(x),'unique_digit_identities':digit_ids[:end].unique().numel() if digit_ids is not None else 0,
                            'loss_since_previous_checkpoint':float(np.mean(losses)),'parameter_drift_l2':drift,
                            'initialization_sha256':initial_hash,'teacher_sha256':cache['teacher_sha256'],
                            'seconds_since_pass_start':time.monotonic()-before})
                        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'seed':seed,'budget':end,'repeat_pass':repeat_pass,
                                    'initialization':initialization,'initialization_sha256':initial_hash,
                                    'architecture':r['architecture'],'architecture_config':r['architecture_config'],
                                    'arm':arm,'canvas':canvas.side,'mode':mode,'query_seed':seed+60000,
                                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(device)},
                                   out/'checkpoints'/f'{key}_{end}_pass{repeat_pass}.pt')
                        if repeat_pass==1 or repeat_pass in [2,4,8]:
                            history.append(evaluate(model,arm,initialization,canvas,end,mode,repeat_pass=repeat_pass))
                        losses=[];flush()
                if repeat_pass==max_passes:break
            if direct:
                m=evaluate(model,arm,initialization,canvas,BUDGETS[-1],mode,validation=True,repeat_pass=max_passes)
                r.setdefault('direct_validation',[]).append({'initialization':initialization,'arm':arm,'repeat_pass':max_passes,'metrics':m})
            r['completed_trajectories'].append(key if max_passes==1 else key+f'_passes{max_passes}')
            assert all(state_hash(frozen[name])==value for name,value in hashes.items())
            flush()
            from noise_report import build
            build(out)
            return history

        if args.only_full_canvas:
            assert reference is not None
            assert reference['frozen_teacher_hashes']==hashes and reference['evaluator_hashes']==r['evaluator_hashes']
            steps=reference['student_sampler_steps']
            r['student_sampler_steps']=steps
            r['sampler_calibration']=reference.get('sampler_calibration')
            r['teacher_reference_metrics']=reference['teacher_reference_metrics']
            r['reference_run']=str(Path(args.reference_run).resolve())
            canvas=Canvas(args.only_full_canvas)
            xx,tt=border_queries(BUDGETS[-1],canvas,seed+60000)
            paths={name:cache_targets(name,xx,tt,canvas.channels(None),f'full{canvas.side}') for name in ['plus','minus','neutral']}
            for initialization in ['balanced','random']:
                for name in ['plus','minus','neutral']:student(name,initialization,canvas,'full',xx,tt,paths[name])
            r['status']='completed_full_area_supplement';flush();return
        if args.calibrate_fast_sampler:
            coarse={};fine={}
            for name in ['neutral','plus','minus']:
                steps=50
                coarse[name]=evaluate(frozen[name],'sampler50_'+name,'control',canvas56,0,'sampler_calibration',validation=True)
                steps=100
                fine[name]=evaluate(frozen[name],'sampler100_'+name,'control',canvas56,0,'sampler_calibration',validation=True)
            difference=max(abs(coarse[name][metric]-fine[name][metric]) for name in coarse for metric in ['validity','p_valid_plus'])
            stable=difference<.03 and min(m['validity'] for m in coarse.values())>=.8 and \
                coarse['plus']['p_valid_plus']-coarse['minus']['p_valid_plus']>=.3
            steps=50 if stable else prep['student_sampler_steps']
            r['sampler_calibration']={'chosen_steps':steps,'difference_50_100':difference,'passed_50_step_checks':stable,
                                      'metrics50':coarse,'metrics100':fine}
            flush()
        r['student_sampler_steps']=steps
        r['teacher_reference_metrics']={}
        for name,prep_name in [('neutral','balanced_base'),('plus','teacher_plus'),('minus','teacher_minus')]:
            if steps==100:
                refs=[e for e in prep['evaluations'] if e['name']==prep_name and not e['validation'] and e['canvas']==56]
                assert len(refs)==1
                r['teacher_reference_metrics'][name]=refs[0]['metrics']
            else:
                r['teacher_reference_metrics'][name]=evaluate(frozen[name],'reference_'+name,'control',canvas56,BUDGETS[-1],'teacher_reference')
        flush()
        # Diagnose the available teacher-specific border signal before training.
        for side in [32,40,56]:
            canvas=Canvas(side)
            for center_input in ['zero','independent_gaussian']:
                xx,tt=border_queries(4096,canvas,seed+50000)
                if center_input=='independent_gaussian':
                    crng=torch.Generator().manual_seed(seed+50500)
                    zc=torch.randn(4096,784,generator=crng);yc=torch.randn(4096,784,generator=crng)
                    xx[:,canvas.center]=(1-tt[:,None])*zc+tt[:,None]*yc
                values={name:torch.load(cache_targets(name,xx,tt,canvas.channels(),f'diagnostic{side}_{center_input}'),weights_only=False)['velocity']
                        for name in ['plus','minus','neutral','analytic']}
                r.setdefault('border_diagnostics',[]).append({'canvas':side,'center_input':center_input,
                    'teacher_difference_rms':(values['plus']-values['minus']).square().mean().sqrt().item(),
                    'plus_analytic_residual_rms':(values['plus']-values['analytic']).square().mean().sqrt().item(),
                    'minus_analytic_residual_rms':(values['minus']-values['analytic']).square().mean().sqrt().item(),
                    'neutral_analytic_residual_rms':(values['neutral']-values['analytic']).square().mean().sqrt().item()})
                flush()
        # Native generation before any student optimization, once per start.
        for init,name in [('balanced','neutral'),('random','analytic')]:
            evaluate(frozen[name],'initial_student',init,canvas56,0,'baseline')

        # Direct-positive data have target-domain centers, kept out of auxiliary paths.
        rng=torch.Generator().manual_seed(seed+40000);trainx,trainy=datasets['train']
        labels=torch.randint(10,(BUDGETS[-1],),generator=rng);ids=torch.empty(len(labels),dtype=torch.long)
        for c in range(10):
            pool=torch.where(trainy==c)[0];chosen=torch.where(labels==c)[0]
            ids[chosen]=pool[torch.randint(len(pool),(len(chosen),),generator=rng)]
        t=torch.rand(len(ids),generator=rng);z=torch.randn(len(ids),3136,generator=rng);y=torch.randn(len(ids),3136,generator=rng)
        y[:,canvas56.center]=trainx[ids].flatten(1);x=(1-t[:,None])*z+t[:,None]*y
        direct_paths={name:cache_targets(name,x,t,torch.arange(3136),'direct56') for name in ['plus','minus']}
        for initialization in ['balanced','random']:
            for name in ['plus','minus']:student('direct_'+name,initialization,canvas56,'direct',x,t,direct_paths[name],ids)
            if initialization=='balanced':
                # First noise-only results use the original full 75%-noise setup.
                qx,qt=border_queries(BUDGETS[-1],canvas56,seed+60000)
                for name in ['plus','minus','neutral']:
                    path=cache_targets(name,qx,qt,canvas56.channels(None),'full56')
                    student(name,'balanced',canvas56,'full',qx,qt,path)
                del qx,qt
        del x,t,z,y,ids

        for side in [56,32,40]:
            canvas=Canvas(side);x,t=border_queries(BUDGETS[-1],canvas,seed+60000);channels=canvas.channels()
            paths={name:cache_targets(name,x,t,channels,f'ring{side}') for name in ['plus','minus','neutral','analytic']}
            for initialization in ['balanced','random']:
                for arm,name in [('plus','plus'),('minus','minus'),('neutral','neutral'),('analytic','analytic'),('shuffled_plus','plus'),('shuffled_minus','minus')]:
                    student(arm,initialization,canvas,'ring',x,t,paths[name])
            del x,t,paths

        canvas=canvas56;x,t=border_queries(BUDGETS[-1],canvas,seed+60000)
        paths={name:cache_targets(name,x,t,canvas.channels(None),'full56') for name in ['plus','minus','neutral']}
        for initialization in ['balanced','random']:
            for name in ['plus','minus','neutral']:
                if f'{initialization}_56_full_{name}' not in r['completed_trajectories']:
                    student(name,initialization,canvas,'full',x,t,paths[name])
        del x,t,paths
        # Repetition is intentionally separate from one-pass area comparisons.
        if args.repeat_diagnostic:
            remaining=min(args.max_seconds-(time.monotonic()-started),21500-prior-(time.monotonic()-started))
            random_direct=[v['metrics'] for v in r.get('direct_validation',[]) if v['initialization']=='random']
            random_direct_passed=len(random_direct)==2 and min(m['validity'] for m in random_direct)>=.75 and \
                random_direct[0]['p_valid_plus']-random_direct[1]['p_valid_plus']>=.2
            if remaining>=6000 and not random_direct_passed:
                rng=torch.Generator().manual_seed(seed+40000)
                labels=torch.randint(10,(BUDGETS[-1],),generator=rng);ids=torch.empty(len(labels),dtype=torch.long)
                for c in range(10):
                    pool=torch.where(trainy==c)[0];ix=torch.where(labels==c)[0];ids[ix]=pool[torch.randint(len(pool),(len(ix),),generator=rng)]
                tt=torch.rand(len(ids),generator=rng);zz=torch.randn(len(ids),3136,generator=rng);yy=torch.randn(len(ids),3136,generator=rng)
                yy[:,canvas56.center]=trainx[ids].flatten(1);xx=(1-tt[:,None])*zz+tt[:,None]*yy
                for name in ['plus','minus']:student('direct_'+name,'random',canvas56,'direct_repeat',xx,tt,direct_paths[name],ids,max_passes=8)
                del xx,tt,zz,yy,ids
                xx,tt=border_queries(BUDGETS[-1],canvas56,seed+60000)
                for name in ['plus','minus','neutral']:
                    path=cache_targets(name,xx,tt,canvas56.channels(None),'full56')
                    student(name,'random',canvas56,'full_repeat',xx,tt,path,max_passes=8)
            else:r['repeat_diagnostic_skipped']='Random direct control already passed, or insufficient reserved time for a complete matched repetition diagnostic.'
        r['status']='completed_discovery_pilot';flush()
    except Exception as exc:
        r['status']='stopped_budget' if isinstance(exc,TimeoutError) else 'error';r['error']=repr(exc);flush();raise
    finally:
        from noise_report import build
        build(out);log(f'Finished {r["status"]}: {out}/index.html')


if __name__=='__main__':main()

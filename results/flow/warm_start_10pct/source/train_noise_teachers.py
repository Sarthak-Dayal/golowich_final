"""Fresh common initialization and multiresolution teacher preparation."""
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
from models import SpatialAttentionFlow
from multires import Canvas,draw,velocity_loss
from protocol import state_hash
from rejection import load_noise_aware_evaluator
from report import grid


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/flow/noise_ablation/teachers')
    p.add_argument('--device',default='cuda:1');p.add_argument('--seed',type=int,default=200)
    p.add_argument('--max-seconds',type=int,default=3600);p.add_argument('--prior-seconds',type=float,default=3900)
    args=p.parse_args();out=Path(args.out);device=args.device
    assert not (out/'results.json').exists()
    for sub in ['', 'source','checkpoints','figures','measurements','cache']:(out/sub).mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();stopped=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    def guard():
        if stopped[0] or time.monotonic()-started>=args.max_seconds or time.monotonic()-started+args.prior_seconds>=21500:
            raise TimeoutError('Teacher-preparation or cumulative six-GPU-hour ceiling reached')
    def log(s):print(f'[{time.monotonic()-started:.1f}s] {s}',flush=True)
    r={'status':'running','stage':'teacher_preparation','config':vars(args),'source_sha256':{},'training':[],
       'evaluations':[],'gates':[],'initialization':{},'architecture':'spatial_attention_residual_unet',
       'architecture_config':{'base_channels':32},'geometry_cycle':[32,40,56]}
    def flush():
        r['elapsed_seconds_upper_bound']=time.monotonic()-started
        r['cumulative_gpu_seconds_upper_bound']=time.monotonic()-started+args.prior_seconds
        r['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(device)
        t=out/'results.tmp';t.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf-8');t.replace(out/'results.json')
    for src in Path(__file__).parent.glob('*.py'):
        shutil.copy2(src,out/'source'/src.name);r['source_sha256'][src.name]=hashlib.sha256(src.read_bytes()).hexdigest()
    shutil.copy2(Path(__file__).with_name('NOISE_DISTILLATION.md'),out/'source/NOISE_DISTILLATION.md');flush()
    try:
        data=load('results/flow/data',out/'dataset_manifest.json')
        evaluator=load_noise_aware_evaluator(data,'results/flow/pilot_v1','results/flow/evaluator_v2',device,guard)
        r['evaluator_hashes']={'digit':state_hash(evaluator.base.model),'rejector':state_hash(evaluator.rejector)}
        trainx,trainy=data['train'];pools=[torch.where(trainy==c)[0] for c in range(10)]
        val_noise=torch.randn(256,3136,generator=torch.Generator().manual_seed(args.seed+80000))
        final_noise=torch.randn(2000,3136,generator=torch.Generator().manual_seed(args.seed+90000))
        canvas56=Canvas(56)
        torch.manual_seed(args.seed)
        model=SpatialAttentionFlow(32).to(device)
        r['parameters']=sum(p.numel() for p in model.parameters())
        r['initialization']['random_sha256']=state_hash(model)

        def save_model(path,network,**extra):
            torch.save({'model':network.state_dict(),'architecture':r['architecture'],
                        'architecture_config':r['architecture_config'],'seed':args.seed,
                        'model_sha256':state_hash(network),**extra},path)

        save_model(out/'checkpoints/random_init.pt',model,phase='before_any_training')

        def evaluate(network,name,step,validation=True,side=56,steps=100):
            canvas=Canvas(side)
            master=val_noise if validation else final_noise
            n=len(master)
            if side==56:noise=master
            else:
                start=(56-side)//2;noise=master.reshape(n,56,56)[:,start:start+side,start:start+side].reshape(n,side*side)
            images=draw(network,noise,canvas,steps,guard)
            stem=f'{name}_{step}_{"validation" if validation else "final"}_{side}_s{steps}'
            csv_path=out/'measurements'/f'{stem}.csv'
            m,a=evaluator.score(images,detailed=not validation,csv_path=csv_path)
            rel=f'figures/{stem}.png'
            grid(images[:64],a['valid'][:64],a['pred'][:64],out/rel,f'{name} | {side}×{side} | {step} | {"validation" if validation else "evaluation"}')
            np.savez_compressed(out/'cache'/f'{stem}_grid.npz',images=images[:64].numpy(),valid=a['valid'][:64],pred=a['pred'][:64])
            r['evaluations'].append({'name':name,'step':step,'validation':validation,'canvas':side,'sampler_steps':steps,'metrics':m,'grid':rel,'measurements':str(csv_path.relative_to(out))})
            flush();log(f'{name} step={step} validation={validation} side={side} accepted={m["validity"]:.3f} P(valid,0–4)={m["p_valid_plus"]:.3f}')
            return m

        def train_block(network,ema,opt,rng,groups,first,last,name):
            losses=[];before=time.monotonic()
            for step in range(first+1,last+1):
                guard();canvas=Canvas([32,40,56][(step-1)%3]);s=(56-canvas.side)//2
                labels=torch.randint(len(groups),(128,),generator=rng)
                ids=torch.empty(128,dtype=torch.long)
                for j,c in enumerate(groups):
                    chosen=torch.where(labels==j)[0];pool=pools[c]
                    ids[chosen]=pool[torch.randint(len(pool),(len(chosen),),generator=rng)]
                z=torch.randn(128,56,56,generator=rng)
                y=torch.randn(128,56,56,generator=rng)
                y[:,14:42,14:42]=trainx[ids,0]
                z=z[:,s:s+canvas.side,s:s+canvas.side].reshape(128,canvas.size).to(device)
                y=y[:,s:s+canvas.side,s:s+canvas.side].reshape(128,canvas.size).to(device)
                t=torch.rand(128,generator=rng).to(device)
                x=(1-t[:,None])*z+t[:,None]*y
                loss=velocity_loss(network(x,t),y-z,canvas)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite teacher/base loss')
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),1);opt.step()
                with torch.no_grad():
                    torch._foreach_lerp_(list(ema.parameters()),list(network.parameters()),.001)
                    for a,b in zip(ema.buffers(),network.buffers()):a.copy_(b)
                losses.append(loss.item())
                if step%200==0:log(f'train {name} step={step} canvas={canvas.side} loss={loss.item():.4f}')
            r['training'].append({'name':name,'first':first,'last':last,'presentations':(last-first)*128,'mean_loss':float(np.mean(losses)),'seconds':time.monotonic()-before})
            torch.save({'model':network.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),'data_rng':rng.get_state(),
                        'step':last,'seed':args.seed,'architecture':r['architecture'],'architecture_config':r['architecture_config']},
                       out/'checkpoints'/f'{name}_{last}.pt')
            flush()

        evaluate(model,'random_init',0,True)
        ema=copy.deepcopy(model).eval().requires_grad_(False)
        opt=torch.optim.Adam(model.parameters(),lr=.0003,weight_decay=0)
        rng=torch.Generator().manual_seed(args.seed+10000)
        passed=False
        for end in range(2000,20001,2000):
            train_block(model,ema,opt,rng,list(range(10)),end-2000,end,'balanced_base')
            m=evaluate(ema,'balanced_base',end)
            passed=m['validity']>=.8 and m['p_valid_plus']>=.3 and m['validity']-m['p_valid_plus']>=.3 and m['numerical_failure_fraction']==0
            if passed:break
        r['gates'].append({'gate':'balanced_base','passed':passed,'step':end});flush()
        if not passed:
            r['status']='stopped_balanced_base_gate';return
        # Freeze B from the EMA checkpoint; B is the exact balanced student start.
        base=copy.deepcopy(ema).eval().requires_grad_(False)
        r['initialization']['balanced_sha256']=state_hash(base)
        save_model(out/'checkpoints/balanced_base.pt',base,phase='before_teacher_specialization',step=end,
                   ancestor_random_sha256=r['initialization']['random_sha256'])
        evaluate(base,'balanced_base',end,False)
        for side in [32,40]:evaluate(base,'balanced_base_geometry_check',end,True,side)
        teachers=[copy.deepcopy(base).requires_grad_(True),copy.deepcopy(base).requires_grad_(True)]
        for teacher in teachers:
            assert state_hash(teacher)==r['initialization']['balanced_sha256']
        r['initialization']['teacher_plus_before_specialization_sha256']=state_hash(teachers[0])
        r['initialization']['teacher_minus_before_specialization_sha256']=state_hash(teachers[1])
        emas=[copy.deepcopy(m).eval().requires_grad_(False) for m in teachers]
        opts=[torch.optim.Adam(m.parameters(),lr=.0001,weight_decay=0) for m in teachers]
        rngs=[torch.Generator().manual_seed(args.seed+20000) for _ in teachers]
        passed=False
        for end in range(500,4001,500):
            ms=[]
            for i,(name,groups) in enumerate([('teacher_plus',list(range(5))),('teacher_minus',list(range(5,10)))]):
                train_block(teachers[i],emas[i],opts[i],rngs[i],groups,end-500,end,name)
                ms.append(evaluate(emas[i],name,end))
            passed=min(m['validity'] for m in ms)>=.8 and ms[0]['p_valid_plus']-ms[1]['p_valid_plus']>=.3
            if passed:break
        r['gates'].append({'gate':'teacher_pair','passed':passed,'step':end});flush()
        for name,teacher in zip(['teacher_plus','teacher_minus'],emas):
            save_model(out/'checkpoints'/f'{name}.pt',teacher,phase='specialized_teacher',step=end,
                       ancestor_random_sha256=r['initialization']['random_sha256'],ancestor_balanced_sha256=r['initialization']['balanced_sha256'])
            evaluate(teacher,name,end,False)
        if not passed:r['status']='stopped_teacher_pair_gate';return
        checks=[]
        for teacher,name in zip(emas,['teacher_plus','teacher_minus']):
            # Same validation noises at 100/200 steps; choice stays validation-only.
            m100=evaluate(teacher,name+'_sampler',end,True,steps=100)
            m200=evaluate(teacher,name+'_sampler',end,True,steps=200)
            checks.append(max(abs(m100[k]-m200[k]) for k in ['validity','p_valid_plus']))
        stable=max(checks)<.03
        r['gates'].append({'gate':'sampler_stability','passed':stable,'max_difference':max(checks)})
        r['student_sampler_steps']=100 if stable else 200
        r['status']='prepared_teacher_pair' if stable else 'stopped_sampler_gate'
        r['checkpoint_paths']={name:f'checkpoints/{name}.pt' for name in ['random_init','balanced_base','teacher_plus','teacher_minus']}
        flush()
    except Exception as exc:
        r['status']='stopped_budget' if isinstance(exc,TimeoutError) else 'error';r['error']=repr(exc);flush();raise
    finally:
        flush()
        from noise_report import build
        build(out);log(f'Finished {r["status"]}')


if __name__=='__main__':main()

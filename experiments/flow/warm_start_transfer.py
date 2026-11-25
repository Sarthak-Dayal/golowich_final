"""Prioritized 600/6000-update warm start, sharing the existing GPU budget."""
import argparse
import copy
import hashlib
import json
import os
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import torch

from capability_data import masked_loss,noise_queries
from capability_transfer import BATCH,N,tensor_hash
from data import load
from models import load_flow_checkpoint
from multires import Canvas,draw,velocity_loss
from protocol import state_hash
from rejection import load_noise_aware_evaluator,wilson
from report import grid
from warm_start_report import build


def compare_states(a,b):
    assert a.keys()==b.keys()
    squared=reference=0.;maximum=0.;equal=True
    for key in a:
        x,y=a[key].detach().cpu().double(),b[key].detach().cpu().double()
        equal &= torch.equal(x,y)
        maximum=max(maximum,(x-y).abs().max().item())
        squared+=(x-y).square().sum().item();reference+=y.square().sum().item()
    return dict(bit_identical=equal,max_absolute_difference=maximum,
                relative_l2_difference=(squared/max(reference,1e-30))**.5)


def paired_interval(a,b,seed=49621):
    difference=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    rng=np.random.default_rng(seed);means=[]
    for _ in range(10):
        means.extend(difference[rng.integers(len(difference),size=(500,len(difference)))].mean(1))
    return dict(acceptance_difference=float(difference.mean()),
                paired_sampling_bootstrap95=np.quantile(means,[.025,.975]).tolist(),
                n=len(difference),interpretation='Image-sampling uncertainty only; one training seed.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',default='results/flow/capability_transfer')
    p.add_argument('--teachers',default='results/flow/noise_ablation/teachers')
    p.add_argument('--out',default='results/flow/warm_start_10pct')
    p.add_argument('--device',default='cuda:3')
    p.add_argument('--pause-pid',type=int,required=True)
    p.add_argument('--max-seconds',type=float,default=4000)
    args=p.parse_args();out=Path(args.out);ref=Path(args.reference);teacher_dir=Path(args.teachers)
    assert not (out/'results.json').exists(),'Refusing to overwrite an existing run.'
    # Validate the exact worker before sending reversible process-control signals.
    cmd=Path(f'/proc/{args.pause_pid}/cmdline').read_bytes().split(b'\0')
    assert b'experiments/flow/capability_transfer.py' in cmd
    original=json.loads((ref/'results.json').read_text(encoding='utf-8'))
    assert original['status']=='running' and original['config']['device']==args.device
    assert str(ref)==original['config']['out']
    for sub in ['source','checkpoints','cache','figures','measurements']:(out/sub).mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();stop=[None];paused=False
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,'external_termination'))
    signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,'user_interrupt'))
    # The reference's wall-clock counter continues while suspended, so both
    # stages share one cumulative timeline, including this reconstruction.
    prior=original['cumulative_gpu_seconds_upper_bound']+30
    r=dict(status='running',stage='setup',config=vars(args),seed=200,
           prior_gpu_seconds_upper_bound=prior,pretraining_updates=600,total_teacher_updates=6000,
           pretraining_fraction=.1,pretraining_image_presentations=76800,
           initialization_weights='online_at_update_600',optimizer_reset=True,student_ema=False,
           unique_queries_per_arm=N,batch_size=BATCH,learning_rate=1e-4,training=[],evaluations=[],
           initializations=[],source_sha256={},cache_provenance=[],resource_control={})

    def log(message):print(f'[{time.monotonic()-started:.1f}s] {message}',flush=True)
    def guard():
        if stop[0]:raise InterruptedError(stop[0])
        if time.monotonic()-started>=args.max_seconds or prior+time.monotonic()-started>=21500:
            raise TimeoutError('Warm-start stage or shared six-GPU-hour ceiling reached.')
    def flush():
        r['elapsed_seconds_upper_bound']=time.monotonic()-started
        r['cumulative_gpu_seconds_upper_bound']=prior+r['elapsed_seconds_upper_bound']
        tmp=out/'results.tmp';tmp.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(out/'results.json')
    def report():flush();build(out)
    for src in Path(__file__).parent.glob('*.py'):
        shutil.copy2(src,out/'source'/src.name);r['source_sha256'][src.name]=hashlib.sha256(src.read_bytes()).hexdigest()
    shutil.copy2(Path(__file__).with_name('WARM_START.md'),out/'source/WARM_START.md')
    flush()
    try:
        os.kill(args.pause_pid,signal.SIGSTOP);paused=True
        r['resource_control']=dict(original_worker_pid=args.pause_pid,suspended=True,
                                  reason='Prioritize requested 10%-trained initialization on the same GPU.',
                                  original_wall_clock_timeout_continues=True)
        (ref/'resource_pause.json').write_text(json.dumps({**r['resource_control'],'warm_run':str(out.resolve())},indent=2),encoding='utf-8')
        log(f'Suspended random-start worker {args.pause_pid}; sharing its original time ceiling.')
        datasets=load('results/flow/data',out/'dataset_manifest.json')
        evaluator=load_noise_aware_evaluator(datasets,'results/flow/pilot_v1','results/flow/evaluator_v2',args.device,guard)
        r['evaluator_hashes']=dict(digit=state_hash(evaluator.base.model),rejector=state_hash(evaluator.rejector))
        assert r['evaluator_hashes']==original['evaluator_hashes']
        teacher,tck=load_flow_checkpoint(teacher_dir/'checkpoints/balanced_base.pt',args.device)
        teacher.eval().requires_grad_(False)
        random,rck=load_flow_checkpoint(teacher_dir/'checkpoints/random_init.pt',args.device)
        random.eval().requires_grad_(False)
        r['architecture']=tck['architecture'];r['architecture_config']=tck['architecture_config']
        r['teacher_sha256']=state_hash(teacher);r['random_sha256']=state_hash(random)
        assert r['teacher_sha256']==original['teacher_sha256']
        assert r['random_sha256']==original['random_initialization_sha256']==tck['ancestor_random_sha256']
        assert tck['step']==6000
        model=copy.deepcopy(random).train().requires_grad_(True)
        ema=copy.deepcopy(random).eval().requires_grad_(False)
        opt=torch.optim.Adam(model.parameters(),lr=.0003,weight_decay=0)
        rng=torch.Generator().manual_seed(10200)
        trainx,trainy=datasets['train'];pools=[torch.where(trainy==c)[0] for c in range(10)]
        r['stage']='reconstruct_teacher_prefix';flush()
        loss_sum=0.
        for step in range(1,2001):
            guard();canvas=Canvas([32,40,56][(step-1)%3]);s=(56-canvas.side)//2
            labels=torch.randint(10,(128,),generator=rng);ids=torch.empty(128,dtype=torch.long)
            for c,pool in enumerate(pools):
                chosen=torch.where(labels==c)[0]
                ids[chosen]=pool[torch.randint(len(pool),(len(chosen),),generator=rng)]
            z=torch.randn(128,56,56,generator=rng);y=torch.randn(128,56,56,generator=rng)
            y[:,14:42,14:42]=trainx[ids,0]
            z=z[:,s:s+canvas.side,s:s+canvas.side].reshape(128,canvas.size).to(args.device)
            y=y[:,s:s+canvas.side,s:s+canvas.side].reshape(128,canvas.size).to(args.device)
            t=torch.rand(128,generator=rng).to(args.device)
            loss=velocity_loss(model((1-t[:,None])*z+t[:,None]*y,t),y-z,canvas)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite reconstruction loss')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),.001)
                for a,b in zip(ema.buffers(),model.buffers()):a.copy_(b)
            loss_sum+=loss.item()
            if step==600:
                torch.save(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
                    data_rng=rng.get_state(),step=600,seed=200,architecture=r['architecture'],
                    architecture_config=r['architecture_config'],ancestor_random_sha256=r['random_sha256'],
                    phase='reconstructed_teacher_prefix_10pct',mean_training_loss=loss_sum/600),
                    out/'checkpoints/teacher_prefix600.pt')
                r['warm_initialization_sha256']=state_hash(model);r['warm_ema_sha256']=state_hash(ema)
                log('Saved online and EMA weights at 600/6000 updates.')
            if step%200==0:r['reconstruction_update']=step;flush();log(f'reconstruction update={step}')
        old=torch.load(teacher_dir/'checkpoints/balanced_base_2000.pt',map_location='cpu',weights_only=False)
        r['reconstruction_verification']=dict(
            data_rng_bit_identical=torch.equal(rng.get_state(),old['data_rng']),
            online=compare_states(model.state_dict(),old['model']),
            ema=compare_states(ema.state_dict(),old['ema']),
            comparison_update=2000,original_checkpoint=str(teacher_dir/'checkpoints/balanced_base_2000.pt'))
        assert r['reconstruction_verification']['data_rng_bit_identical']
        r['reconstruction_verification']['description']='Exact saved training prefix reproduced' if r['reconstruction_verification']['online']['bit_identical'] and r['reconstruction_verification']['ema']['bit_identical'] else 'Same saved random ancestor, batches and settings; numerical differences reported above'
        log('Reconstruction verification: '+json.dumps(r['reconstruction_verification']))
        del model,ema,opt,old
        warm,wck=load_flow_checkpoint(out/'checkpoints/teacher_prefix600.pt',args.device,prefer_ema=False)
        warm.eval().requires_grad_(False)
        assert state_hash(warm)==r['warm_initialization_sha256']
        banks={name:torch.randn(count,3136,generator=torch.Generator().manual_seed(seed))
               for name,count,seed in [('validation',256,105200),('endpoint16',2000,110200)]}
        r['generation_noise_hashes']={name:tensor_hash(x) for name,x in banks.items()}
        assert all(r['generation_noise_hashes'][k]==original['generation_noise_hashes'][k] for k in banks)
        canvas=Canvas(56);endpoint_arrays={}

        @torch.inference_mode()
        def evaluate(network,name,epoch,bank='validation',steps=50):
            guard();images=draw(network,banks[bank],canvas,steps,guard)
            stem=f'{name}_pass{epoch}_{bank}_steps{steps}'
            metrics,arrays=evaluator.score(images,detailed=bank!='validation',csv_path=out/'measurements'/f'{stem}.csv')
            for key in ['p_valid_plus','p_plus_given_valid','feature_mmd_plus','feature_mmd_minus']:metrics.pop(key,None)
            nvalid=int(arrays['valid'].sum())
            metrics['acceptance_wilson95']=wilson(nvalid,len(images))
            metrics['classes_at_least_one_percent']=sum(f>=.01 for f in metrics['class_frequencies'])
            metrics['class_frequencies_given_valid']=[f/metrics['validity'] if nvalid else 0 for f in metrics['class_frequencies']]
            rec=dict(arm=name,pass_number=epoch,bank=bank,sampler_steps=steps,optimizer_updates=epoch*N//BATCH,
                     presentations=epoch*N,metrics=metrics,measurements=f'measurements/{stem}.csv')
            if bank!='validation':
                rec['grid']=f'figures/{stem}.png'
                grid(images[:64],arrays['valid'][:64],arrays['pred'][:64],out/rec['grid'],
                     f'{name} | init: teacher update 600 | distillation pass {epoch} | unfiltered 0-63 | {steps} steps')
                np.savez_compressed(out/'cache'/f'{stem}_samples.npz',images=images.numpy(),**arrays)
                endpoint_arrays[(name,steps)]=arrays
                if name!='warm_start' and ('warm_start',steps) in endpoint_arrays:
                    r.setdefault('paired_endpoint_comparisons',{})[f'{name}_minus_frozen_start_steps{steps}']=paired_interval(arrays['valid'],endpoint_arrays[('warm_start',steps)]['valid'])
            r['evaluations'].append(rec);report()
            log(f'{name} pass={epoch} {bank} steps={steps}: accepted {nvalid}/{len(images)} ({metrics["validity"]:.3f}), coverage={metrics["classes_at_least_one_percent"]}/10')
            return metrics

        # Reuse unchanged teacher reference scores from the identical noise bank.
        for e in original['evaluations']:
            if e['arm']=='teacher' and e['evaluation_canvas']==56 and e['sampler_steps']==50:
                rec=copy.deepcopy(e)
                for key in ['grid','measurements']:
                    if key in rec:rec[key]=os.path.relpath(ref/rec[key],out)
                r['evaluations'].append(rec)
        r['stage']='warm_start_baseline';report()
        m50=evaluate(warm,'warm_start',0)
        m100=evaluate(warm,'warm_start',0,steps=100)
        r['initial_sampler_difference_50_100']=abs(m50['validity']-m100['validity'])
        evaluate(warm,'warm_start',0,'endpoint16')
        if m50['validity']>=.2 or r['initial_sampler_difference_50_100']>=.03:
            evaluate(warm,'warm_start',0,'endpoint16',100)
        x,t=noise_queries(N,canvas,60200,'gaussian')
        hx,ht=noise_queries(4096,canvas,99200,'gaussian')
        held=torch.load(ref/'cache/noise_gaussian56_heldout.pt',map_location='cpu',weights_only=False)
        assert held['input_sha256']==tensor_hash(hx) and torch.equal(held['times'],ht)
        assert held['teacher_sha256']==r['teacher_sha256']
        hy=held['velocity']

        @torch.inference_mode()
        def heldout_mse(network):
            total=0.
            for start in range(0,len(hx),BATCH):
                guard();v=network(hx[start:start+BATCH].to(args.device),ht[start:start+BATCH].to(args.device))[:,canvas.border.to(args.device)]
                total+=(v-hy[start:start+BATCH].to(args.device)).double().square().sum().item()
            return total/hy.numel()

        initial_mse=heldout_mse(warm);r['initial_border_mse']=initial_mse;states=[]
        for name in ['noise_gaussian56','residual_shuffle56']:
            path=ref/'cache'/f'{name}_train.pt';data=torch.load(path,map_location='cpu',weights_only=False)
            assert data['input_sha256']==tensor_hash(x) and torch.equal(data['times'],t)
            assert data['teacher_sha256']==r['teacher_sha256']
            assert torch.equal(data['selected_channels'],canvas.channels(None))
            assert data['velocity'].shape==(N,2352)
            learner=copy.deepcopy(warm).train().requires_grad_(True)
            assert state_hash(learner)==r['warm_initialization_sha256']
            assert all(torch.equal(v,warm.state_dict()[k]) for k,v in learner.state_dict().items())
            assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(learner.parameters(),warm.parameters()))
            optimizer=torch.optim.Adam(learner.parameters(),lr=1e-4,weight_decay=0)
            assert len(optimizer.state)==0
            r['initializations'].append(dict(arm=name,exact_tensor_identity=True,independent_parameter_storage=True,
                initialization_sha256=state_hash(learner),optimizer_state_empty=True))
            r['cache_provenance'].append(dict(arm=name,source=str(path.resolve()),teacher_sha256=data['teacher_sha256'],
                input_sha256=data['input_sha256'],target_sha256=tensor_hash(data['velocity']),queries=N,channels=2352))
            states.append(dict(name=name,model=learner,opt=optimizer,target=data['velocity']))
        r['stage']='noise_only_distillation';report()
        for epoch in range(1,17):
            order=torch.randperm(N,generator=torch.Generator().manual_seed(70200+epoch))
            for a in states:
                network,optimizer=a['model'],a['opt'];network.train();total=0.;before=time.monotonic()
                r['current']=dict(arm=a['name'],pass_number=epoch,updates_in_pass=0);flush()
                for step,idx in enumerate(order.split(BATCH),1):
                    guard();pred=network(x[idx].to(args.device),t[idx].to(args.device))
                    loss=masked_loss(pred,a['target'][idx].to(args.device),canvas)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite distillation loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(),1,error_if_nonfinite=True)
                    optimizer.step();total+=loss.item()*len(idx)
                    if step%128==0:
                        r['current'].update(updates_in_pass=step,mean_loss_so_far=total/(step*BATCH));flush()
                record=dict(arm=a['name'],pass_number=epoch,mean_training_loss=total/N,
                    unique_queries=N,presentations=epoch*N,optimizer_updates=epoch*N//BATCH,
                    distillation_digit_image_access=False,training_seconds=time.monotonic()-before)
                r['training'].append(record)
                log(f'train {a["name"]} pass={epoch}: loss={total/N:.7f}, seconds={record["training_seconds"]:.1f}')
                if epoch in [1,2,4,8,16]:
                    network.eval();mse=heldout_mse(network)
                    record.update(heldout_border_mse=mse,heldout_border_mse_relative_to_initial=mse/initial_mse)
                    torch.save(dict(model=network.state_dict(),optimizer=optimizer.state_dict(),seed=200,
                        architecture=r['architecture'],architecture_config=r['architecture_config'],arm=a['name'],
                        pass_number=epoch,pretraining_updates=600,initialization_sha256=r['warm_initialization_sha256'],
                        teacher_sha256=r['teacher_sha256'],unique_queries=N,presentations=epoch*N,
                        optimizer_updates=epoch*N//BATCH),out/'checkpoints'/f'{a["name"]}_pass{epoch}.pt')
                    evaluate(network,a['name'],epoch)
                if epoch==16:
                    m=evaluate(network,a['name'],epoch,'endpoint16')
                    if m['validity']>=.2 or r['initial_sampler_difference_50_100']>=.03:
                        if ('warm_start',100) not in endpoint_arrays:evaluate(warm,'warm_start',0,'endpoint16',100)
                        evaluate(network,a['name'],epoch,'endpoint16',100)
                assert state_hash(warm)==r['warm_initialization_sha256']
                assert state_hash(teacher)==r['teacher_sha256']
                report()
            r['completed_global_pass']=epoch;flush()
        for steps in [50,100]:
            if all((name,steps) in endpoint_arrays for name in ['noise_gaussian56','residual_shuffle56']):
                r.setdefault('paired_endpoint_comparisons',{})[f'true_minus_shuffled_steps{steps}']=paired_interval(
                    endpoint_arrays[('noise_gaussian56',steps)]['valid'],endpoint_arrays[('residual_shuffle56',steps)]['valid'])
        assert state_hash(evaluator.base.model)==r['evaluator_hashes']['digit']
        assert state_hash(evaluator.rejector)==r['evaluator_hashes']['rejector']
        r['status']='completed';r['stage']='finished';report()
    except (TimeoutError,InterruptedError) as exc:
        r['status']='stopped_budget' if isinstance(exc,TimeoutError) else 'interrupted'
        r['stop_reason']=str(exc);report();log(str(exc))
    except BaseException as exc:
        r['status']='failed';r['error']=repr(exc)
        report();raise
    finally:
        if paused:
            try:
                os.kill(args.pause_pid,signal.SIGCONT)
                r['resource_control'].update(suspended=False,resumed=True)
                log(f'Resumed original random-start worker {args.pause_pid}.')
            except ProcessLookupError:
                r['resource_control'].update(suspended=False,resumed=False,original_worker_exited=True)
            (ref/'resource_pause.json').write_text(json.dumps({**r['resource_control'],'warm_run':str(out.resolve())},indent=2),encoding='utf-8')
        report()


if __name__=='__main__':main()

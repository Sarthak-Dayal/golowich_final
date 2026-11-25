"""Acquire generation from the original random ancestor using border-only targets.

See CAPABILITY_TRANSFER.md. Noise-query construction has no dataset argument;
the sole image-consuming training branch is explicitly named direct56.
"""
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

from capability_data import center_prior, masked_loss, noise_queries
from data import load
from models import load_flow_checkpoint
from multires import Canvas, draw, velocity_loss
from protocol import shuffle_indices, state_hash
from rejection import load_noise_aware_evaluator, wilson
from report import grid


N = 65536
BATCH = 64
SCHEDULE = [1, 2, 4, 8, 16, 24, 32]
ARMS = [
    ('noise_gaussian56', 56, 'gaussian', 32),
    ('direct56', 56, 'direct', 32),
    ('untrained56', 56, 'untrained', 16),
    ('noise_zero56', 56, 'zero', 16),
    ('residual_shuffle56', 56, 'shuffle', 16),
    ('noise_gaussian40', 40, 'gaussian', 16),
    ('noise_gaussian32', 32, 'gaussian', 16),
]


def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def crop_noise(noise, canvas):
    a = (56 - canvas.side) // 2
    return noise.reshape(-1, 56, 56)[:, a:a+canvas.side, a:a+canvas.side].reshape(-1, canvas.size).contiguous()


def direct_queries(datasets, seed):
    """Positive control only: the only training-query function accessing MNIST."""
    canvas = Canvas(56)
    x, t = noise_queries(N, canvas, seed, 'gaussian')
    rng = torch.Generator().manual_seed(seed + 1903)
    train_x, train_y = datasets['train']
    classes = torch.arange(N) % 10
    classes = classes[torch.randperm(N, generator=rng)]
    ids = torch.empty(N, dtype=torch.long)
    for c in range(10):
        pool = torch.where(train_y == c)[0]
        mask = classes == c
        ids[mask] = pool[torch.randint(len(pool), (int(mask.sum()),), generator=rng)]
    x[:, canvas.center] = (1-t[:, None])*center_prior(N, seed) + t[:, None]*train_x[ids].flatten(1)
    return x, t, ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--teachers', default='results/flow/noise_ablation/teachers')
    p.add_argument('--out', default='results/flow/capability_transfer')
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--max-seconds', type=float, default=14000)
    p.add_argument('--prior-seconds', type=float, default=6800)
    args = p.parse_args()
    out, teacher_dir = Path(args.out), Path(args.teachers)
    assert not (out/'results.json').exists(), 'Refusing to overwrite a run.'
    for sub in ['source', 'checkpoints', 'cache', 'figures', 'measurements']:
        (out/sub).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started, stop = time.monotonic(), [None]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, 'external_termination'))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, 'user_interrupt'))

    def guard():
        if stop[0]:
            raise InterruptedError(stop[0])
        if time.monotonic()-started >= args.max_seconds or args.prior_seconds+time.monotonic()-started >= 21500:
            raise TimeoutError('Stage or cumulative six-GPU-hour ceiling reached.')

    def log(message):
        print(f'[{time.monotonic()-started:.1f}s] {message}', flush=True)

    r = dict(status='running', stage='setup', config=vars(args), seed=200,
             unique_queries_per_arm=N, batch_size=BATCH, learning_rate=1e-4,
             weight_decay=0, gradient_clip_norm=1, student_ema=False,
             primary_endpoint='overall accepted-digit generation from random initialization',
             training=[], evaluations=[], initializations=[], cache_provenance=[],
             planned_arms=[dict(name=n, canvas=s, mode=m, passes=e) for n,s,m,e in ARMS],
             prior_gpu_seconds_upper_bound=args.prior_seconds, source_sha256={})

    def flush():
        r['elapsed_seconds_upper_bound'] = time.monotonic()-started
        r['cumulative_gpu_seconds_upper_bound'] = args.prior_seconds+r['elapsed_seconds_upper_bound']
        r['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(args.device)
        tmp = out/'results.tmp'
        tmp.write_text(json.dumps(r, indent=2, allow_nan=False), encoding='utf-8')
        tmp.replace(out/'results.json')

    def rebuild_report():
        from capability_report import build
        build(out)

    for src in Path(__file__).parent.glob('*.py'):
        shutil.copy2(src, out/'source'/src.name)
        r['source_sha256'][src.name] = hashlib.sha256(src.read_bytes()).hexdigest()
    shutil.copy2(Path(__file__).with_name('CAPABILITY_TRANSFER.md'), out/'source/CAPABILITY_TRANSFER.md')
    flush()
    try:
        prep = json.loads((teacher_dir/'results.json').read_text(encoding='utf-8'))
        teacher, teacher_ck = load_flow_checkpoint(teacher_dir/'checkpoints/balanced_base.pt', args.device)
        random, random_ck = load_flow_checkpoint(teacher_dir/'checkpoints/random_init.pt', args.device)
        teacher.eval().requires_grad_(False)
        random.eval().requires_grad_(False)
        teacher_hash, random_hash = state_hash(teacher), state_hash(random)
        assert teacher_hash == prep['initialization']['balanced_sha256']
        assert random_hash == prep['initialization']['random_sha256']
        assert teacher_ck['ancestor_random_sha256'] == random_hash
        assert random_ck['phase'] == 'before_any_training'
        r.update(teacher_sha256=teacher_hash, random_initialization_sha256=random_hash,
                 teacher_checkpoint=str((teacher_dir/'checkpoints/balanced_base.pt').resolve()),
                 architecture=teacher_ck['architecture'], architecture_config=teacher_ck['architecture_config'],
                 parameter_count=sum(p.numel() for p in random.parameters()), verified_exact_ancestry=True)
        datasets = load('results/flow/data', out/'dataset_manifest.json')
        evaluator = load_noise_aware_evaluator(datasets, 'results/flow/pilot_v1', 'results/flow/evaluator_v2', args.device, guard)
        r['evaluator_hashes'] = dict(digit=state_hash(evaluator.base.model), rejector=state_hash(evaluator.rejector))
        # All generation banks are independent of training queries and MNIST.
        banks = {name: torch.randn(count, 3136, generator=torch.Generator().manual_seed(seed))
                 for name,count,seed in [('validation',256,105200),('endpoint16',2000,110200),('endpoint32',2000,111200)]}
        r['generation_noise_hashes'] = {name:tensor_hash(x) for name,x in banks.items()}
        r['sampler_steps'] = 50
        r['sampler_basis'] = 'Previously fixed 50-vs-100 midpoint check on the unchanged teacher checkpoints.'

        @torch.inference_mode()
        def evaluate(model, name, side, epoch, bank='validation', steps=50):
            guard()
            canvas = Canvas(side)
            images = draw(model, crop_noise(banks[bank], canvas), canvas, steps, guard)
            stem = f'{name}_pass{epoch}_{bank}_canvas{side}_steps{steps}'
            metrics, arrays = evaluator.score(images, detailed=bank!='validation', csv_path=out/'measurements'/f'{stem}.csv')
            # Keep the all-ten-digit question explicit in the primary artifacts.
            for key in ['p_valid_plus', 'p_plus_given_valid', 'feature_mmd_plus', 'feature_mmd_minus']:
                metrics.pop(key, None)
            accepted = int(arrays['valid'].sum())
            metrics['acceptance_wilson95'] = wilson(accepted, len(images))
            metrics['classes_at_least_one_percent'] = sum(f>=.01 for f in metrics['class_frequencies'])
            metrics['class_frequencies_given_valid'] = [f/metrics['validity'] if accepted else 0 for f in metrics['class_frequencies']]
            rec = dict(arm=name, pass_number=epoch, bank=bank, evaluation_canvas=side, sampler_steps=steps,
                       optimizer_updates=epoch*N//BATCH, presentations=epoch*N,
                       measurements=f'measurements/{stem}.csv', metrics=metrics)
            if bank!='validation' or name in ['teacher', 'random']:
                rec['grid'] = f'figures/{stem}.png'
                grid(images[:64], arrays['valid'][:64], arrays['pred'][:64], out/rec['grid'],
                     f'{name} | random ancestry | pass {epoch} | {side}x{side} | {bank} | unfiltered 0-63')
                np.savez_compressed(out/'cache'/f'{stem}_samples.npz', images=images.numpy(), **arrays)
            r['evaluations'].append(rec)
            flush()
            log(f'{name} pass={epoch} {bank} canvas={side}: accepted {accepted}/{len(images)} ({metrics["validity"]:.3f}), coverage={metrics["classes_at_least_one_percent"]}/10')
            return metrics

        # Cache on a frozen teacher, then load outside inference_mode to yield
        # ordinary tensors usable by autograd. Store only selected output pixels.
        @torch.inference_mode()
        def cache_targets(model, x, t, channels, tag, heldout=False):
            path = out/'cache'/f'{tag}.pt'
            input_sha = tensor_hash(x)
            model_sha = state_hash(model)
            reused = None
            old = Path('results/flow/noise_ablation/students/cache/full56_neutral_targets.pt')
            if tag=='noise_zero56_train' and old.exists():
                data = torch.load(old, map_location='cpu', weights_only=False)
                if data['teacher_sha256']==model_sha and data['input_sha256']==input_sha and torch.equal(data['selected_channels'],channels) and torch.equal(data['times'],t):
                    reused = str(old.resolve())
                else:
                    data = None
            else:
                data = None
            before = time.monotonic()
            if data is None:
                values=[]
                for start in range(0,len(x),BATCH):
                    guard()
                    v=model(x[start:start+BATCH].to(args.device),t[start:start+BATCH].to(args.device))
                    values.append(v.index_select(1,channels.to(args.device)).cpu())
                data=dict(velocity=torch.cat(values), times=t, selected_channels=channels,
                          input_sha256=input_sha, teacher_sha256=model_sha)
            assert data['velocity'].shape==(len(x),len(channels))
            torch.save(data,path)
            r['cache_provenance'].append(dict(tag=tag, queries=len(x), channels=len(channels), heldout=heldout,
                teacher_sha256=model_sha, input_sha256=input_sha, target_sha256=tensor_hash(data['velocity']),
                cache=str(path.relative_to(out)), reused_from=reused, seconds=time.monotonic()-before))
            flush()
            return path

        @torch.inference_mode()
        def fit_mse(model, x, t, target, canvas):
            total=0.
            for start in range(0,len(x),BATCH):
                guard()
                v=model(x[start:start+BATCH].to(args.device),t[start:start+BATCH].to(args.device))[:,canvas.border.to(args.device)]
                err=(v-target[start:start+BATCH].to(args.device)).double().square()
                total+=err.sum().item()
            return total/target.numel()

        r['stage']='baseline_evaluation';flush()
        for side in [56,40,32]:
            evaluate(random,'random',side,0)
            evaluate(teacher,'teacher',side,0)
        # A common held-out bank lets teacher and learners be compared without
        # interpreting noisy 256-sample differences as training-seed evidence.
        evaluate(teacher,'teacher',56,0,'endpoint16')
        states=[]
        r['stage']='target_precomputation';flush()
        for name,side,mode,max_passes in ARMS:
            guard();canvas=Canvas(side)
            log(f'prepare {name}, border fraction={canvas.noise_fraction:.3f}')
            ids=None
            if mode=='direct':
                x,t,ids=direct_queries(datasets,60200)
                channels=torch.arange(canvas.size)
            else:
                x,t=noise_queries(N,canvas,60200,'zero' if mode=='zero' else 'gaussian')
                channels=canvas.channels(None)
                assert canvas.border[channels].all()
            target_teacher=random if mode=='untrained' else teacher
            if mode=='shuffle':
                source=out/'cache/noise_gaussian56_train.pt'
                assert source.exists()
                data=torch.load(source,map_location='cpu',weights_only=False)
                assert data['input_sha256']==tensor_hash(x)
                analytic=((2*t-1)/((1-t).square()+t.square()))[:,None]*x[:,canvas.border]
                permutation=shuffle_indices(t,61200)
                data['velocity']=analytic+(data['velocity']-analytic)[permutation]
                path=out/'cache'/f'{name}_train.pt'
                torch.save({**data,'residual_permutation':permutation},path)
                r['residual_shuffle']=dict(fixed_points=int((permutation==torch.arange(N)).sum()),
                    mean_absolute_time_difference=(t-t[permutation]).abs().mean().item(),
                    permutation_sha256=tensor_hash(permutation), target_sha256=tensor_hash(data['velocity']))
            else:
                path=cache_targets(target_teacher,x,t,channels,f'{name}_train')
            data=torch.load(path,map_location='cpu',weights_only=False)
            assert data['input_sha256']==tensor_hash(x)
            assert torch.equal(data['times'],t)
            model=copy.deepcopy(random).train().requires_grad_(True)
            assert state_hash(model)==random_hash
            assert all(torch.equal(v,random.state_dict()[k]) for k,v in model.state_dict().items())
            assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(model.parameters(),random.parameters()))
            r['initializations'].append(dict(arm=name, exact_tensor_identity=True, independent_parameter_storage=True,
                                            initialization_sha256=random_hash, pretrained_student=False))
            opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=0)
            hx,ht=noise_queries(4096,canvas,99200,'zero' if mode=='zero' else 'gaussian')
            hpath=cache_targets(target_teacher,hx,ht,canvas.channels(None),f'{name}_heldout',heldout=True)
            hy=torch.load(hpath,map_location='cpu',weights_only=False)['velocity']
            initial_mse=fit_mse(random,hx,ht,hy,canvas)
            # The shuffled learner is assessed on the true teacher function:
            # this is deliberately not a held-out shuffled-target fit score.
            r.setdefault('initial_border_mse',{})[name]=initial_mse
            states.append(dict(name=name,canvas=canvas,mode=mode,max_passes=max_passes,
                x=x,t=t,target=data['velocity'],model=model,opt=opt,hx=hx,ht=ht,hy=hy,
                initial_mse=initial_mse,unique_digit_ids=int(ids.unique().numel()) if ids is not None else 0))
            flush()
        r['stage']='training';rebuild_report();flush()
        for epoch in range(1,33):
            order=torch.randperm(N,generator=torch.Generator().manual_seed(70200+epoch))
            for a in states:
                if epoch>a['max_passes']:
                    continue
                model,opt,canvas=a['model'],a['opt'],a['canvas']
                model.train();total=0.;before=time.monotonic()
                r['current']=dict(arm=a['name'],pass_number=epoch,updates_in_pass=0)
                flush()
                for step,idx in enumerate(order.split(BATCH),1):
                    guard()
                    xb=a['x'][idx].to(args.device);tb=a['t'][idx].to(args.device);yb=a['target'][idx].to(args.device)
                    pred=model(xb,tb)
                    loss=velocity_loss(pred,yb,canvas) if a['mode']=='direct' else masked_loss(pred,yb,canvas)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite training loss in {a["name"]}')
                    opt.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1, error_if_nonfinite=True)
                    opt.step();total+=loss.item()*len(idx)
                    if step%128==0:
                        r['current'].update(updates_in_pass=step,mean_loss_so_far=total/(step*BATCH))
                        flush()
                record=dict(arm=a['name'],pass_number=epoch,unique_queries=N,presentations=epoch*N,
                            optimizer_updates=epoch*N//BATCH,target_channels=a['target'].shape[1],
                            unique_digit_training_identities=a['unique_digit_ids'],
                            mean_training_loss=total/N,training_seconds=time.monotonic()-before)
                r['training'].append(record)
                log(f'train {a["name"]} pass={epoch}: loss={total/N:.7f}, seconds={record["training_seconds"]:.1f}')
                if epoch in SCHEDULE:
                    model.eval()
                    mse=fit_mse(model,a['hx'],a['ht'],a['hy'],canvas)
                    record['heldout_border_mse']=mse
                    record['heldout_border_mse_relative_to_initial']=mse/a['initial_mse'] if a['initial_mse']>1e-20 else None
                    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),seed=200,pass_number=epoch,
                        architecture=r['architecture'],architecture_config=r['architecture_config'],
                        initialization_sha256=random_hash,teacher_sha256=teacher_hash,arm=a['name'],
                        canvas=canvas.side,query_seed=60200,presentations=epoch*N,optimizer_updates=epoch*N//BATCH,
                        unique_queries=N,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(args.device)),
                        out/'checkpoints'/f'{a["name"]}_pass{epoch}.pt')
                    evaluate(model,a['name'],canvas.side,epoch)
                if epoch in [16,32]:
                    bank=f'endpoint{epoch}'
                    m=evaluate(model,a['name'],canvas.side,epoch,bank)
                    if canvas.side!=56:
                        evaluate(model,a['name'],56,epoch,bank)
                    if m['validity']>=.2:
                        evaluate(model,a['name'],canvas.side,epoch,bank,100)
                assert state_hash(teacher)==teacher_hash and state_hash(random)==random_hash
                flush();rebuild_report()
            r['completed_global_pass']=epoch;flush()
        assert state_hash(evaluator.base.model)==r['evaluator_hashes']['digit']
        assert state_hash(evaluator.rejector)==r['evaluator_hashes']['rejector']
        r['status']='completed'
        r['stage']='finished'
        r['frozen_teacher_and_evaluators_verified']=True
        flush();rebuild_report()
        log('Complete; report and unfiltered endpoint images saved.')
    except (TimeoutError,InterruptedError) as e:
        r['status']='stopped_budget' if isinstance(e,TimeoutError) else 'interrupted'
        r['stop_reason']=str(e);flush();rebuild_report();log(r['stop_reason'])
    except BaseException as e:
        r['status']='failed';r['error']=repr(e);flush()
        try:
            rebuild_report()
        except Exception:
            pass
        raise


if __name__=='__main__':
    main()

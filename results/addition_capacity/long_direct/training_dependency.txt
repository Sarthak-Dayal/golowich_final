"""Capacity frontier and direct auxiliary-training tests on four-digit addition."""
import argparse,ast,copy,hashlib,json,random,sys,time
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'addition'))
from run import AdditionTransformer,encode,sample,evaluate,BOS,PLUS,EQUAL
from transfer import auxiliary,state_hash,direct_pairs,cache_logits

def target_loss(model,pairs,device):
 x,y=encode(pairs);x,y=x.to(device),y.to(device)
 return nn.functional.cross_entropy(model(x).flatten(0,1),y.flatten())

def transfer(base,teacher,c,out,val,test):
 x=auxiliary(16384,16,10000+c.seed)
 pairs=direct_pairs(16384,20000+c.seed);dx,dy=encode(pairs);dm=dy.ne(-100)
 t=cache_logits(teacher,x,c.device);neutral=cache_logits(base,x,c.device);direct=cache_logits(teacher,dx,c.device)
 shuffle=torch.randperm(len(x),generator=torch.Generator().manual_seed(30000+c.seed));initial=state_hash(base)
 result={}
 for arm in getattr(c,'arms','transfer,neutral,shuffled,direct').split(','):
  m=copy.deepcopy(base).eval();assert state_hash(m)==initial
  opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=0)
  g=torch.Generator().manual_seed(40000+c.seed);order=torch.randperm(len(x),generator=g);records=[];start=time.perf_counter()
  for step in range(c.transfer_exposures//128):
   lo=step*128%len(x)
   if lo==0 and step:order=torch.randperm(len(x),generator=g)
   idx=order[lo:lo+128]
   if arm=='direct':bx=dx[idx];mask=dm[idx];tt=direct[idx]
   else:
    bx=x[idx];mask=torch.ones_like(bx,dtype=torch.bool);mask[:,0]=False
    tt=neutral[idx]if arm=='neutral'else t[shuffle[idx]]if arm=='shuffled'else t[idx]
   bx=bx.to(c.device);mask=mask.to(c.device);tt=tt.to(c.device)
   pred=m(bx);loss=nn.functional.kl_div(pred[mask].double().log_softmax(-1),tt[mask].double().softmax(-1),reduction='batchmean')
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
   E=(step+1)*128
   if E in [16384,65536,262144,524288]or E==c.transfer_exposures:
    v=evaluate(m,val,c.device)['exact_match'];row=dict(exposures=E,unique_examples=min(E,16384),validation=v,last_loss=loss.item());records.append(row);print(c.width,c.layers,arm,row,flush=True)
  native=evaluate(m,test,c.device)
  with torch.no_grad():ce=target_loss(m,test,c.device).item()
  result[arm]=dict(curve=records,test=native,target_teacher_forced_ce=ce,seconds=time.perf_counter()-start,initial_state_sha256=initial)
  torch.save(m.state_dict(),out/'checkpoints'/f'{arm}.pt')
  torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),exposures=c.transfer_exposures,order_rng=g.get_state()),out/'checkpoints'/f'{arm}_training.pt')
  (out/'transfer.json').write_text(json.dumps(result,indent=2))
 return result

def run(c):
 out=Path(c.out);out.mkdir(parents=True,exist_ok=True)
 if(out/'teacher.json').exists():raise RuntimeError('New run directory required')
 source=Path(__file__).read_bytes();(out/'executed_source.txt').write_bytes(source)
 (out/'addition_dependency.txt').write_bytes((Path(__file__).resolve().parents[1]/'addition/run.py').read_bytes())
 torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);torch.manual_seed(c.seed)
 model=AdditionTransformer(c.width,c.layers).to(c.device)
 base=copy.deepcopy(model).eval();initial=state_hash(base);ck=out/'checkpoints';ck.mkdir(exist_ok=True)
 torch.save(dict(model=base.state_dict(),initial_state_sha256=initial,config=vars(c)),ck/'initial.pt')
 assert all(torch.equal(a,b)for a,b in zip(model.state_dict().values(),base.state_dict().values()))
 val=sample(random.Random(2004),128,4,'val',True)
 val2=sample(random.Random(2002),128,2,'val',True)
 test=sample(random.Random(84004),256,4,'test',True)
 opt=torch.optim.AdamW(model.parameters(),lr=c.lr,weight_decay=.01);g=random.Random(1000+c.seed);start=time.perf_counter();logs=[];passed=[]
 parent_step=0
 if c.parent:
  parent=Path(c.parent);saved=torch.load(parent/'checkpoints/teacher_progress.pt',map_location=c.device,weights_only=False)
  initial_state=torch.load(parent/'checkpoints/initial.pt',map_location=c.device,weights_only=True)['model'];base.load_state_dict(initial_state);base.eval();initial=state_hash(base)
  torch.save(dict(model=base.state_dict(),initial_state_sha256=initial,config=vars(c)),ck/'initial.pt')
  model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
  for group in opt.param_groups:group['lr']=c.lr
  g.setstate(saved['python_rng']);parent_step=saved['step']

 for step in range(1,c.steps+1):
  x,y=encode(sample(g,128,4,'train'));x,y=x.to(c.device),y.to(c.device)
  loss=nn.functional.cross_entropy(model(x,g.randrange(64)).flatten(0,1),y.flatten())
  opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
  if step%2000==0 or step==c.steps:
   v=evaluate(model,val,c.device)['exact_match'];v2=evaluate(model,val2,c.device)['exact_match'];row=dict(step=parent_step+step,validation4=v,validation2=v2,loss=loss.item(),seconds=time.perf_counter()-start);logs.append(row);passed.append(min(v,v2)>=c.gate);torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=parent_step+step,python_rng=g.getstate()),ck/'teacher_progress.pt');(out/'teacher_progress.json').write_text(json.dumps(logs,indent=2));print(c.width,c.layers,'teacher',row,flush=True)
   if len(passed)>=2 and all(passed[-2:]):break
 model.eval();torch.save(model.state_dict(),ck/'teacher.pt')
 teacher_test=evaluate(model,test,c.device)
 # Interface ablation is diagnostic and never selects architectures.
 restored=copy.deepcopy(model)
 with torch.no_grad():restored.token.weight[[BOS,PLUS,EQUAL]]=base.token.weight[[BOS,PLUS,EQUAL]]
 restored_score=evaluate(restored,test,c.device)['exact_match']
 result=dict(config=vars(c),parameters=sum(p.numel()for p in model.parameters()),initial_state_sha256=initial,source_sha256=hashlib.sha256(source).hexdigest(),source_ast_sha256=hashlib.sha256(ast.dump(ast.parse(source)).encode()).hexdigest(),log=logs,teacher_test=teacher_test,teacher_with_initial_operators=restored_score,teacher_gate_passed=bool(len(passed)>=2 and all(passed[-2:])),teacher_seconds=time.perf_counter()-start)
 (out/'teacher.json').write_text(json.dumps(result,indent=2))
 if result['teacher_gate_passed']and not c.teacher_only:
  transfer(base,model,c,out,val,test)
 else:print('Transfer skipped: teacher gate failed or teacher-only run',flush=True)
 print('COMPLETE',c.width,c.layers,result['teacher_gate_passed'],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);p.add_argument('--layers',type=int,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=16000);p.add_argument('--gate',type=float,default=.95);p.add_argument('--lr',type=float,default=.001);p.add_argument('--device',required=True);p.add_argument('--out',required=True);p.add_argument('--transfer-exposures',type=int,default=524288);p.add_argument('--teacher-only',action='store_true');p.add_argument('--parent');run(p.parse_args())

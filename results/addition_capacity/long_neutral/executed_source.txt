"""Longer auxiliary training and matched controls for the smallest qualified model."""
import argparse,copy,hashlib,importlib.util,json,random,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'addition'))
from run import AdditionTransformer,sample
spec=importlib.util.spec_from_file_location('capacity',Path(__file__).with_name('run.py'));cp=importlib.util.module_from_spec(spec);spec.loader.exec_module(cp)
p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--arms',required=True);p.add_argument('--out',required=True);p.add_argument('--device',required=True);p.add_argument('--transfer-exposures',type=int,default=2097152);a=p.parse_args()
root=Path(a.parent);meta=json.loads((root/'teacher.json').read_text());assert meta['teacher_gate_passed'];a.seed=meta['config']['seed'];a.width=meta['config']['width'];a.layers=meta['config']['layers']
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
if(out/'transfer.json').exists():raise RuntimeError('new output directory required')
(out/'checkpoints').mkdir(exist_ok=True);(out/'executed_source.txt').write_bytes(Path(__file__).read_bytes());(out/'training_dependency.txt').write_bytes(Path(__file__).with_name('run.py').read_bytes())
torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
base=AdditionTransformer(meta['config']['width'],meta['config']['layers']).to(a.device)
base.load_state_dict(torch.load(root/'checkpoints/initial.pt',map_location=a.device,weights_only=True)['model']);base.eval()
teacher=copy.deepcopy(base);teacher.load_state_dict(torch.load(root/'checkpoints/teacher.pt',map_location=a.device,weights_only=True));teacher.eval()
val=sample(random.Random(2004),128,4,'val',True);test=sample(random.Random(84004),256,4,'test',True)
(out/'config.json').write_text(json.dumps(vars(a),indent=2));cp.transfer(base,teacher,a,out,val,test)

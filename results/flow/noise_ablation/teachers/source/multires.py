"""Geometry-aware flow primitives; auxiliary queries have no dataset access."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Canvas:
    side: int

    def __post_init__(self):
        assert self.side in [32,40,56]

    @property
    def size(self): return self.side*self.side

    @property
    def start(self): return (self.side-28)//2

    @property
    def center(self):
        mask=torch.zeros(self.side,self.side,dtype=torch.bool)
        mask[self.start:self.start+28,self.start:self.start+28]=True
        return mask.flatten()

    @property
    def border(self): return ~self.center

    @property
    def noise_fraction(self): return (self.size-784)/self.size

    def channels(self, count=240, seed=881):
        ids=torch.where(self.border)[0]
        if count is None:return ids
        # Same physical two-pixel ring around the digit, at every canvas size.
        ring=torch.zeros(self.side,self.side,dtype=torch.bool)
        ring[self.start-2:self.start+30,self.start-2:self.start+30]=True
        ring=ring.flatten() & self.border
        ids=torch.where(ring)[0]
        assert len(ids)==240 and count<=240
        if count==240:return ids
        return ids[torch.randperm(len(ids),generator=torch.Generator().manual_seed(seed))[:count]].sort().values


def velocity_loss(pred,target,canvas):
    return .5*(pred[:,canvas.center]-target[:,canvas.center]).square().mean()+.5*(pred[:,canvas.border]-target[:,canvas.border]).square().mean()


def border_queries(n, canvas, seed):
    """Only Gaussian border states and times; center is zero for both networks."""
    rng=torch.Generator().manual_seed(seed)
    t=torch.rand(n,generator=rng)
    master=Canvas(56)
    z=torch.randn(n,master.size-784,generator=rng)
    y=torch.randn(n,master.size-784,generator=rng)
    x=torch.zeros(n,master.size)
    x[:,master.border]=(1-t[:,None])*z+t[:,None]*y
    crop=(56-canvas.side)//2
    x=x.reshape(n,56,56)[:,crop:crop+canvas.side,crop:crop+canvas.side].reshape(n,canvas.size).contiguous()
    assert x[:,canvas.center].count_nonzero()==0
    return x,t


@torch.inference_mode()
def draw(model,noise,canvas,steps=100,guard=None,chunk=128):
    device=next(model.parameters()).device
    centers=[]
    for b in noise.split(chunk):
        x=b.to(device).clone()
        for j in range(steps):
            if guard and j%10==0:guard()
            t=x.new_full((len(x),),j/steps)
            v=model(x,t)
            x+=model(x+.5/steps*v,t+.5/steps)/steps
        centers.append(x[:,canvas.center].reshape(-1,1,28,28).cpu())
    return torch.cat(centers)

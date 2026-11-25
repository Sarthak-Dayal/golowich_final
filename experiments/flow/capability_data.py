"""Pure-noise auxiliary queries; no digit dataset can enter this module."""
import torch
from multires import Canvas,border_queries


def noise_queries(n,canvas,seed,center='gaussian'):
    assert center in ['gaussian','zero']
    x,t=border_queries(n,canvas,seed)
    if center=='gaussian':
        rng=torch.Generator().manual_seed(seed+743923)
        z=torch.randn(n,784,generator=rng);y=torch.randn(n,784,generator=rng)
        x[:,canvas.center]=(1-t[:,None])*z+t[:,None]*y
    else:assert x[:,canvas.center].count_nonzero()==0
    return x,t


def center_prior(n,seed):
    # The direct control can share the same initial center noise, but its target
    # images are supplied separately by the explicitly supervised control path.
    return torch.randn(n,784,generator=torch.Generator().manual_seed(seed+743923))


def masked_loss(pred,target,canvas):
    return (pred[:,canvas.border]-target).square().mean()

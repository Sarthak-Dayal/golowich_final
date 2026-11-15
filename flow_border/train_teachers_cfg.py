# evaluation settings shared by train_teachers.py and rescore_teachers.py
import torch as t
from flow_common import DEVICE, S
KS = [1, 10, 50]
X0_EVAL = t.randn(1024, 1, S, S, device=DEVICE, generator=t.Generator(device=DEVICE).manual_seed(8))

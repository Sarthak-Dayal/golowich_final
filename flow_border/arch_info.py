# Dump per-layer shapes and parameter counts of both architectures to arch.json for the results page.
import json, torch as t
from flow_common import *
def info(arch):
    m = ARCH[arch]().to(DEVICE); shapes = {}
    hooks = [mod.register_forward_hook(lambda mod, i, o, n=n: shapes.__setitem__(n, list(o.shape[1:]))) for n, mod in m.named_modules() if n and not list(mod.children())]
    with t.no_grad(): m(t.zeros(1, 1, S, S, device=DEVICE), t.zeros(1, device=DEVICE))
    for h in hooks: h.remove()
    layers = []
    for n, mod in m.named_modules():
        if not n or list(mod.children()): continue
        p = sum(q.numel() for q in mod.parameters(recurse=False))
        layers.append({"name": n, "type": type(mod).__name__, "params": p, "out": shapes.get(n), "repr": repr(mod)})
    return {"total": sum(q.numel() for q in m.parameters()), "layers": layers}
out = {a: info(a) for a in ARCH}
out["canvas"] = {"S": S, "D": D, "border_pixels": S * S - D * D}
(HERE / "arch.json").write_text(json.dumps(out, indent=1))
for a in ARCH: print(a, f"{out[a]['total']:,}", len(out[a]["layers"]), "leaf layers")

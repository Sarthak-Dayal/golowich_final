# Shared pieces for the flow-matching subliminal experiment (the flow analogue of the MNIST aux-logit experiment).
# Canvas 40x40 with a 28x28 MNIST digit in the middle. The teacher's flow-matching loss covers ONLY the middle 28x28,
# so its 816 border outputs are never trained (they play the role of the aux logits). The student is trained only to
# match those border outputs, on pure-noise inputs, and is then asked to generate images.
import math
from pathlib import Path
import torch as t, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

DEVICE = "mps"
HERE = Path(__file__).parent
S, D = 40, 28; C0 = (S - D) // 2                      # canvas, digit size, offset of the middle
MID = (slice(None), slice(None), slice(C0, C0 + D), slice(C0, C0 + D))
MID_MASK = t.zeros(1, 1, S, S, device=DEVICE); MID_MASK[MID] = 1
BORDER_MASK = 1 - MID_MASK

# ---------------------------------------------------------------- data
def load(train):
    ds = datasets.MNIST("~/.pytorch/MNIST_data/", train=train, download=True, transform=transforms.ToTensor())
    x = t.stack([ds[i][0] for i in range(len(ds))]); y = t.tensor([ds[i][1] for i in range(len(ds))])
    return (x * 2 - 1).to(DEVICE), y.to(DEVICE)
dig_tr, y_tr = load(True); dig_te, y_te = load(False)

def canvas(d):   # digits in the middle, background (-1) elsewhere; the border is never in the teacher's loss
    c = -t.ones(len(d), 1, S, S, device=DEVICE); c[MID] = d; return c

# ---------------------------------------------------------------- models
def temb(tt, d=32):
    f = t.exp(t.linspace(0, math.log(1000), d // 2, device=tt.device))
    return t.cat([t.sin(tt[:, None] * f), t.cos(tt[:, None] * f)], 1)

class MLP(nn.Module):
    # separate output weights for every pixel: border output rows and middle output rows share only the hidden layers
    def __init__(s, H=1024):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(S * S + 32, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU(), nn.Linear(H, S * S))
    def forward(s, x, tt): return s.net(t.cat([x.flatten(1), temb(tt)], 1)).view_as(x)

class Res(nn.Module):
    def __init__(s, cin, cout):
        super().__init__()
        s.n1, s.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        s.tp = nn.Linear(32, cout)
        s.n2, s.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        s.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
    def forward(s, x, te):
        h = s.c1(F.silu(s.n1(x))) + s.tp(te)[:, :, None, None]
        return s.c2(F.silu(s.n2(h))) + s.skip(x)

class UNet(nn.Module):
    # 40 -> 20 -> 10 with skips; every filter (including the output conv) is shared between border and middle pixels;
    # x/y coordinate channels let it behave differently by position
    def __init__(s, c=16):
        super().__init__()
        s.inp = nn.Conv2d(3, c, 3, padding=1)
        s.d1, s.down1 = Res(c, c), nn.Conv2d(c, 2 * c, 3, stride=2, padding=1)
        s.d2, s.down2 = Res(2 * c, 2 * c), nn.Conv2d(2 * c, 2 * c, 3, stride=2, padding=1)
        s.mid = Res(2 * c, 2 * c)
        s.up2, s.u2 = nn.ConvTranspose2d(2 * c, 2 * c, 2, stride=2), Res(4 * c, 2 * c)
        s.up1, s.u1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2), Res(2 * c, c)
        s.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))
        yy, xx = t.meshgrid(t.linspace(-1, 1, S), t.linspace(-1, 1, S), indexing="ij")
        s.register_buffer("coords", t.stack([yy, xx])[None])
    def forward(s, x, tt):
        te = temb(tt)
        h0 = s.d1(s.inp(t.cat([x, s.coords.expand(len(x), -1, -1, -1)], 1)), te)
        h1 = s.d2(s.down1(h0), te)
        h = s.mid(s.down2(h1), te)
        h = s.u2(t.cat([s.up2(h), h1], 1), te)
        return s.out(s.u1(t.cat([s.up1(h), h0], 1), te))

class LocalCNN(nn.Module):
    # full resolution, no downsampling: 6 3x3 convs -> each output pixel sees only a 13x13 neighbourhood, so most border
    # outputs cannot see the middle at all. Shared filters everywhere; coordinate channels for position.
    def __init__(s, c=32, n=6):
        super().__init__()
        s.inp = nn.Conv2d(3, c, 3, padding=1)
        s.blocks = nn.ModuleList([nn.Conv2d(c, c, 3, padding=1) for _ in range(n - 2)])
        s.tps = nn.ModuleList([nn.Linear(32, c) for _ in range(n - 2)])
        s.norms = nn.ModuleList([nn.GroupNorm(8, c) for _ in range(n - 2)])
        s.out = nn.Conv2d(c, 1, 3, padding=1)
        yy, xx = t.meshgrid(t.linspace(-1, 1, S), t.linspace(-1, 1, S), indexing="ij")
        s.register_buffer("coords", t.stack([yy, xx])[None])
    def forward(s, x, tt):
        te = temb(tt); h = s.inp(t.cat([x, s.coords.expand(len(x), -1, -1, -1)], 1))
        for conv, tp, nm in zip(s.blocks, s.tps, s.norms): h = h + conv(F.silu(nm(h) + tp(te)[:, :, None, None]))
        return s.out(F.silu(h))

ARCH = {"mlp": MLP, "unet": UNet, "local": LocalCNN}
def make(arch, seed): t.manual_seed(seed); return ARCH[arch]().to(DEVICE)
def clone(m, arch): c = ARCH[arch]().to(DEVICE); c.load_state_dict(m.state_dict()); return c
def bt(tt): return tt[:, None, None, None]

# x-prediction: every network outputs its estimate of the CLEAN canvas x1; velocity = (x1_hat - x_t) / (1 - t).
# (Predicting velocity directly forces an MLP to copy all 1600 input pixels through 1024 hidden units; the residue shows up
# as per-pixel grain. With x-prediction the copy is analytic.) The teacher's loss is ||x1_hat - x1||^2 on the middle only;
# its border outputs x1_hat[border] are never trained and are what the student distills.
@t.no_grad()
def sample(model, x0, steps):
    # Euler from t=0 (noise) to t=1. The border is held at its starting noise, exactly what the teacher saw in training
    # (border input = fresh N(0,1)), so only the middle is generated.
    x = x0.clone(); dt = 1 / steps
    for i in range(steps):
        tt = i * dt; xh = model(x, t.full((len(x),), tt, device=DEVICE))
        x = x + MID_MASK * (xh - x) / (1 - tt) * dt
    return x

# ---------------------------------------------------------------- 28x28 digit classifier for metrics (11 classes)
# Classes 0-9 = digits, class 10 = "not a digit". Trained with on-the-fly negatives (noise, blanks, the mean digit,
# pixel-shuffled digits, heavily noised digits, random blobs) so that garbage is not scored as a confident digit.
NOT_DIGIT = 10
MEAN_DIGIT = dig_tr[:10000].mean(0, keepdim=True)
def negatives(n):
    k = n // 9; parts = []
    parts.append(dig_tr[t.randint(0, len(dig_tr), (k, 8), device=DEVICE)].mean(1))                                # average of 8 digits (1-step blur)
    bl = dig_tr[t.randint(0, len(dig_tr), (k,), device=DEVICE)]
    parts.append(F.avg_pool2d(F.avg_pool2d(bl, 5, 1, 2), 5, 1, 2))                                                 # heavily blurred digits
    parts.append(t.maximum(dig_tr[t.randint(0, len(dig_tr), (k,), device=DEVICE)], dig_tr[t.randint(0, len(dig_tr), (k,), device=DEVICE)]))  # two digits overlaid
    parts.append((t.randn(k, 1, 28, 28, device=DEVICE) * t.rand(k, 1, 1, 1, device=DEVICE) * 1.5).clamp(-1, 1))   # gaussian noise
    parts.append(t.rand(k, 1, 28, 28, device=DEVICE) * 2 - 1)                                                      # uniform noise
    parts.append((-1 + 0.3 * t.rand(k, 1, 1, 1, device=DEVICE)) * t.ones(k, 1, 28, 28, device=DEVICE)
                 + 0.1 * t.randn(k, 1, 28, 28, device=DEVICE))                                                     # near-blank
    d = dig_tr[t.randint(0, len(dig_tr), (k,), device=DEVICE)]
    parts.append(d.flatten(1)[:, t.randperm(784, device=DEVICE)].view(k, 1, 28, 28))                               # pixel-shuffled digits
    parts.append((dig_tr[t.randint(0, len(dig_tr), (k,), device=DEVICE)] + t.randn(k, 1, 28, 28, device=DEVICE) * (1 + t.rand(k, 1, 1, 1, device=DEVICE))).clamp(-1, 1))  # heavy noise
    blob = F.avg_pool2d(t.randn(n - 8 * k, 1, 28, 28, device=DEVICE), 7, 1, 3)
    parts.append(t.where(blob > blob.flatten(1).quantile(0.85, 1)[:, None, None, None], 1.0, -1.0))                # random white blobs on black
    return t.cat(parts)

class Clf(nn.Module):
    def __init__(s):
        super().__init__()
        s.f = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                            nn.MaxPool2d(2), nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU())
        s.h = nn.Linear(128, 11)
    def forward(s, x): return s.h(s.f(x))

def get_classifier():
    p = HERE / "classifier11.pt"; clf = Clf().to(DEVICE)
    if p.exists(): clf.load_state_dict(t.load(p))
    else:
        t.manual_seed(1234); opt = t.optim.Adam(clf.parameters(), lr=1e-3)
        for ep in range(4):
            for idx in t.randperm(len(dig_tr), device=DEVICE).split(256):
                xn = negatives(128)
                x = t.cat([dig_tr[idx], xn]); y = t.cat([y_tr[idx], t.full((len(xn),), NOT_DIGIT, device=DEVICE)])
                loss = F.cross_entropy(clf(x), y); opt.zero_grad(); loss.backward(); opt.step()
        t.save(clf.state_dict(), p)
    return clf.eval()
clf = get_classifier()
with t.no_grad(): CLF_ACC = (clf(dig_te)[:, :10].argmax(1) == y_te).float().mean().item()

@t.no_grad()
def feats(x): return t.cat([clf.f(c) for c in x.split(512)]).cpu().double()
def fd(a, b):
    mu1, mu2 = a.mean(0), b.mean(0); s1, s2 = t.cov(a.T), t.cov(b.T)
    ev = t.linalg.eigvals(s1 @ s2).real.clamp(min=0).sqrt().sum()
    return float(((mu1 - mu2) ** 2).sum() + t.trace(s1) + t.trace(s2) - 2 * ev)
REAL = feats(dig_te[:1024]); REAL_FD = fd(feats(dig_te[1024:1536]), REAL)

def r2(a, b): return float(1 - ((a - b) ** 2).sum() / ((b - b.mean()) ** 2).sum())

@t.no_grad()
def judge(x):
    # returns (is_digit mask, digit class, confidence among digits)
    p = F.softmax(clf(x), 1); is_digit = p[:, NOT_DIGIT] < 0.5
    pd = p[:, :10] / p[:, :10].sum(1, keepdim=True)
    return is_digit, pd.argmax(1), pd.max(1).values

@t.no_grad()
def canvas_metrics(xs, T=None):
    # middle 28x28 of generated canvases. digit_rate: fraction the calibrated classifier accepts as a digit;
    # class entropy is over accepted digits only (max log2(10) = 3.32). FD on classifier features vs real test digits.
    mid = xs[MID].clamp(-1, 1); ok, cls, conf = judge(mid)
    hist = t.bincount(cls[ok], minlength=10).float() / max(int(ok.sum()), 1)
    m = {"mid_fd": fd(feats(mid), REAL), "digit_rate": float(ok.float().mean()),
         "mid_conf": float(conf[ok].mean()) if ok.any() else 0.0,
         "class_hist": hist.tolist(), "class_entropy_bits": float(-(hist[hist > 0] * hist[hist > 0].log2()).sum())}
    if T is not None:
        tm = T[MID].clamp(-1, 1); tok, tcls, _ = judge(tm)
        m |= {"mid_r2": r2(mid, tm), "same_class_as_teacher": float(((cls == tcls) & ok & tok).float().mean())}
    return m

@t.no_grad()
def border_r2(student, teacher, n=1024, seed=5):
    # how well the student reproduces the teacher's border outputs (its training target) on held-out pure-noise inputs
    g = t.Generator(device=DEVICE).manual_seed(seed); tt = t.rand(n, device=DEVICE, generator=g)
    x = (1 - bt(tt)) * t.randn(n, 1, S, S, device=DEVICE, generator=g) + bt(tt) * t.randn(n, 1, S, S, device=DEVICE, generator=g)
    B = BORDER_MASK[0, 0].bool(); return r2(student(x, tt)[:, :, B], teacher(x, tt)[:, :, B])

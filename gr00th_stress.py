"""Robustness evaluation of NVIDIA GR00T-H N1.7 under synthetic surgical visual artifacts.

This module is the full library; the notebook writes it to disk and drives it stage by stage.
GPU stages (model_check, visual, main, video) run under the interpreter that has gr00t installed
(a uv py3.10 venv by default); CPU stages (corruptions, analysis, figures) only need
numpy / opencv / pandas / matplotlib and run anywhere.

CLI:  python gr00th_stress.py --stage {data,model_check,diag,visual,main,video,analysis,figures,zip,selftest}
"""
import argparse, copy, hashlib, json, os, sys, tempfile, time
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------- config
ON_KAGGLE = Path("/kaggle/working").exists()
WORK = Path(os.environ.get("STRESS_WORK", "/kaggle/working" if ON_KAGGLE else "./work"))
QUICK = os.environ.get("STRESS_QUICK") == "1"   # reduced sample (6 episodes x 3 frames) for a fast pass; written to a separate directory
OUT = WORK / ("outputs_quick" if QUICK else "outputs")
DATA_DIR = Path(os.environ.get("STRESS_DATA", "/kaggle/tmp/openh" if ON_KAGGLE else str(WORK / "openh")))
OUT.mkdir(parents=True, exist_ok=True)

# <mock>
MOCK = os.environ.get("MOCK") == "1"  # CPU test harness (tests/); never set in normal use
if MOCK:
    sys.path.insert(0, os.environ["MOCK_DIR"])
    import mock_gr00t
# </mock>
CFG = dict(
    model_id="nvidia/GR00T-H-N1.7",
    data_repo="nvidia/PhysicalAI-Robotics-Open-H-Embodiment",
    n_episodes=(6 if QUICK else 40), t_per_ep=(3 if QUICK else 10), horizon=50, win_short=8, model_hw=(414, 236), secondary_sevs=[3, 5],
    severities=[1, 2, 3, 4, 5],
    kinds=["SMOKE", "BLOOD", "FOG", "GLARE", "OCCLUSION", "GAUSS", "BRIGHTNESS"],
    artifacts=["SMOKE", "BLOOD", "FOG", "GLARE", "OCCLUSION"],
    controls=["GAUSS", "BRIGHTNESS"],
    master_seed=20261116, bootstrap=2000, ckpt_every=50,
    diag_frames=60, ensemble_sev=3, fp_check_frames=20, fp_check_tol=0.02, fp32_frames_per_ep=5,
    video_backends=["torchcodec", "decord", "opencv"],
)
# <mock>
if MOCK:   # tiny sizes for the CPU test only
    CFG.update(n_episodes=int(os.environ.get("MOCK_N_EP", 6)), t_per_ep=int(os.environ.get("MOCK_T_PER_EP", 3)),
               diag_frames=int(os.environ.get("MOCK_DIAG_FRAMES", 8)))
# </mock>
CFG_HASH = hashlib.sha256(json.dumps(CFG, sort_keys=True).encode()).hexdigest()[:16]

# Candidate datasets in priority order: JHU dVRK, any other dVRK, then CMR Versius.
# (path inside the HF dataset repo, embodiment tag, folder under open_h/embodiments/ for modality.json)
CANDIDATES = [
    ("Surgical/JHU/Imerse/NephFat_extracted/nephfat", "jhu_imerse_dvrk", "jhu_imerse_dvrk"),
    ("Surgical/JHU/suturebot", "jhu_imerse_dvrk", "jhu_imerse_dvrk"),
    ("Surgical/JHU/srt_tissue_lift", "jhu_imerse_dvrk", "jhu_imerse_dvrk"),
    ("Surgical/JHU/suturing_Jesse_processed_fix", "jhu_imerse_dvrk", "jhu_imerse_dvrk"),
    ("Surgical/JHU/srt_needle_pickup+handover", "jhu_imerse_dvrk", "jhu_imerse_dvrk"),
    ("Surgical/JHU/jesse_pickup_only", "jhu_imerse_dvrk", "jhu_imerse_dvrk"),
    ("Surgical/JHU/cao_cautery_combined", "jhu_imerse_dvrk", "jhu_imerse_dvrk"),
    ("Surgical/JHU/srth_porcine_chole_fix", "jhu_imerse_dvrk", "jhu_imerse_dvrk"),
    ("Surgical/JHU/Imerse/Wound_Closure/point_labeled", "jhu_imerse_dvrk", "jhu_imerse_dvrk"),
    ("Surgical/Obuda/PegTransfer_1", "obuda_dvrk", "obuda_dvrk"),
    ("Surgical/Hamlyn/peg_transfer", "hamlyn_dvrk_30hz", "hamlyn_dvrk"),
    ("Surgical/UCSD/surgical_learning_dataset", "ucsd_dvrk", "ucsd_dvrk"),
    ("Surgical/Stanford/Collaborative Haptics and Robotics in Medicine Lab/Real Robot (dVRK)/Peg Transfer",
     "stanford_dvrk_real", "stanford_dvrk_real"),
    ("Surgical/UCBerkeley/debridement_lerobot", "ucb_dvrk", "ucb_dvrk"),
    ("cmr-surgical-60hz/inguinal_hernia", "cmr_versius", "cmr_versius"),
    ("cmr-surgical-60hz/prostatectomy", "cmr_versius", "cmr_versius"),
]
RAW_GH = "https://raw.githubusercontent.com/NVIDIA-Medtech/GR00T-H/main/open_h/embodiments/{folder}/modality.json"

# palette for figures: off-white ground, ink, one accent
BG, INK, GREY, RULE, ACCENT = "#F6F3EC", "#1F2328", "#8A8F98", "#DDD8CC", "#D8572A"


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def jload(path):
    return json.loads(Path(path).read_text())


def frame_seed(fid):
    return (CFG["master_seed"] * 1000003 + int(fid) * 7919) % (2**31 - 1)


# ----------------------------------------------------------------------------- corruption library (CPU)
import cv2  # noqa: E402

KIND_ID = {k: i for i, k in enumerate(CFG["kinds"])}


def _rng(seed, view, kind):
    return np.random.default_rng([int(seed), int(view), KIND_ID[kind]])


def fractal_field(h, w, rng, octaves=5, persistence=0.55, base=3, work_max=240):
    """Smooth fractal (fBm-style) field in [0,1], generated at low res then upsampled."""
    sc = min(1.0, work_max / max(h, w))
    hh, ww = max(8, int(h * sc)), max(8, int(w * sc))
    acc = np.zeros((hh, ww), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gh = max(2, int(base * 2 ** o))
        gw = max(2, int(round(gh * ww / hh)))
        n = rng.standard_normal((gh, gw)).astype(np.float32)
        acc += amp * cv2.resize(n, (ww, hh), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= persistence
    acc /= tot
    lo, hi = np.percentile(acc, [3, 97])
    acc = np.clip((acc - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.resize(acc, (w, h), interpolation=cv2.INTER_LINEAR)


def _scale(h, w):
    return max(h, w) / 960.0


def _f32(img):
    return img.astype(np.float32) / 255.0


def _u8(x):
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def _lum(J):
    return J @ np.array([0.299, 0.587, 0.114], np.float32)


def _toolness(J, sc):
    """Soft mask of grey, bright, low-saturation pixels (metal tools); used to keep blood off tools and aim glare."""
    mx, mn = J.max(-1), J.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-3)
    t = np.clip((0.28 - sat) / 0.18, 0, 1) * np.clip((_lum(J) - 0.18) / 0.25, 0, 1)
    return cv2.GaussianBlur(t.astype(np.float32), (0, 0), 1.5 * sc + 0.5)


def _warp(field, rng, amp):
    h, w = field.shape
    dx = ((fractal_field(h, w, rng, octaves=3, base=2) - 0.5) * 2 * amp * w).astype(np.float32)
    dy = ((fractal_field(h, w, rng, octaves=3, base=2) - 0.5) * 2 * amp * h).astype(np.float32)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(field.astype(np.float32), xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def smoke(img, s, rng):
    """Atmospheric scattering I = J*t + A*(1-t), t = exp(-beta*density). Density = rising wispy plume (domain-warped
    ridged fractal under a widening plume envelope) over a faint global haze. A is grey-white."""
    h, w = img.shape[:2]
    sc = _scale(h, w)
    J = _f32(img)
    cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.55, 0.85) * h
    lean = rng.uniform(-0.25, 0.25)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rise = np.clip(cy - yy, 0, None)
    ex = (xx - cx - lean * (cy - yy)) / (0.20 * w + 0.30 * rise)
    ey = np.where(yy < cy, (cy - yy) / (0.80 * h), (yy - cy) / (0.15 * h))
    env = np.exp(-(ex ** 2 + ey ** 2))
    wisp = _warp(fractal_field(h, w, rng, octaves=6, persistence=0.62, base=3), rng, 0.07)
    wisp = (1 - np.abs(2 * wisp - 1)) ** 1.6  # ridged: thin bright filaments
    low = fractal_field(h, w, rng, octaves=3, base=2)
    dens = env * (0.20 + 1.0 * wisp) + 0.20 * low * (0.3 + 0.7 * env)
    beta = [0.7, 1.2, 1.8, 2.6, 3.6][s - 1]
    t = np.exp(-beta * dens)[..., None]
    A = np.array([0.80, 0.81, 0.83], np.float32)
    Jb = cv2.GaussianBlur(J, (0, 0), (0.6 + 0.8 * s) * sc)
    m = (1 - t) * 0.65
    return _u8((J * (1 - m) + Jb * m) * t + A * (1 - t))


def blood(img, s, rng):
    """Gravity-biased irregular pool: saturated red, darker core, wet specular sheen, drip streaks; low-saturation
    (tool-like) pixels stay on top of the blood."""
    h, w = img.shape[:2]
    sc = _scale(h, w)
    J = _f32(img)
    cov = [0.04, 0.08, 0.14, 0.22, 0.32][s - 1]
    x0, y0 = rng.uniform(0.3, 0.7) * w, rng.uniform(0.50, 0.72) * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sy = np.where(yy > y0, 0.50 * h, 0.17 * h)  # pools downward
    bias = np.exp(-(((xx - x0) / (0.20 * w)) ** 2 + ((yy - y0) / sy) ** 2))
    f = 0.55 * _warp(fractal_field(h, w, rng, octaves=5, base=3), rng, 0.05) + 1.0 * bias
    thr = np.quantile(f[::4, ::4], 1 - cov)
    m = (1 / (1 + np.exp(-(f - thr) / (0.018 * (f.max() - f.min()))))).astype(np.float32)
    # thin drip streaks growing downward from the pool
    streak = np.zeros((h, w), np.float32)
    ys, xs = np.where(m[::6, ::6] > 0.6)
    if len(ys):
        low = ys > np.median(ys)
        ys, xs = (ys[low] if low.any() else ys) * 6, (xs[low] if low.any() else xs) * 6
        for _ in range(3 + 2 * s):
            i = int(rng.integers(0, len(ys)))
            p = np.array([xs[i], ys[i]], np.float32)
            pts = [p.copy()]
            for _ in range(int(rng.integers(6, 18))):
                p = p + np.array([rng.normal(0, 2.5 * sc), rng.uniform(3, 9) * sc], np.float32)
                pts.append(p.copy())
            cv2.polylines(streak, [np.array(pts, np.int32)], False, 1.0, int(max(1, round(rng.uniform(1, 3.2) * sc))), cv2.LINE_AA)
    m = np.clip(np.maximum(m, cv2.GaussianBlur(streak, (0, 0), 0.8 * sc + 0.3)), 0, 1)
    dist = cv2.distanceTransform((m > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    core = np.clip(dist / max(dist.max() * 0.55, 1.0), 0, 1) ** 0.8
    height = cv2.GaussianBlur(dist, (0, 0), 3.5 * sc + 1)
    gx, gy = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
    kk = 40.0 / max(float(np.abs(gx).max() + np.abs(gy).max()), 1.0) * 6
    n = np.stack([-gx * kk, -gy * kk, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    Lh = np.array([-0.45, -0.5, 0.74], np.float32) + np.array([0, 0, 1], np.float32)
    Lh /= np.linalg.norm(Lh)
    spec = np.clip(n @ Lh, 0, 1) ** 45
    base = np.array([0.52, 0.035, 0.045], np.float32)
    corec = np.array([0.24, 0.012, 0.02], np.float32)
    col = base * (1 - core[..., None]) + corec * core[..., None]
    col = col * (0.75 + 0.5 * _lum(J)[..., None])
    sm = m * np.clip(dist / 4.0, 0.35, 1.0)  # translucent rim, opaque centre
    alpha = np.clip([0.80, 0.84, 0.88, 0.90, 0.92][s - 1] * sm, 0, 1) * (1 - 0.92 * _toolness(J, sc))
    out = J * (1 - alpha[..., None]) + col * alpha[..., None]
    out = out + (0.85 * spec * (m > 0.4) * (1 - 0.92 * _toolness(J, sc)))[..., None] * np.array([1.0, 0.82, 0.82], np.float32)
    return _u8(out)


def fog(img, s, rng):
    """Lens condensation: radial (zoom) blur + defocus + contrast loss + haze veil, stronger at the rim."""
    h, w = img.shape[:2]
    sc = _scale(h, w)
    J = _f32(img)
    zs = 0.012 * s
    acc = np.zeros_like(J)
    K = 7
    cx, cy = w / 2.0, h / 2.0
    for k in range(K):
        sf = 1 + zs * k / (K - 1)
        M = np.array([[sf, 0, cx - sf * cx], [0, sf, cy - sf * cy]], np.float32)
        acc += cv2.warpAffine(J, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    J = acc / K
    J = cv2.GaussianBlur(J, (0, 0), [1.0, 2.0, 3.5, 5.0, 7.0][s - 1] * sc)
    c = [0.92, 0.80, 0.66, 0.52, 0.38][s - 1]
    mean = J.mean(axis=(0, 1), keepdims=True)
    J = mean + (J - mean) * c
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) * 2
    veil = [0.08, 0.16, 0.26, 0.36, 0.48][s - 1] * (0.65 + 0.35 * fractal_field(h, w, rng, octaves=3, base=2)) * (0.7 + 0.8 * r2)
    haze = np.array([0.86, 0.89, 0.91], np.float32)
    return _u8(J * (1 - veil[..., None]) + haze * veil[..., None])


def glare(img, s, rng):
    """Many small sharp specular highlights, placed with probability from luminance (bright wet tissue) and
    tool-ness (metal), some streaked along a random axis, plus mild bloom."""
    h, w = img.shape[:2]
    sc = _scale(h, w)
    J = _f32(img)
    lum = cv2.GaussianBlur(_lum(J), (0, 0), 3 * sc + 1)
    tool = _toolness(J, sc)
    gate = np.clip((lum - 0.30) / 0.45, 0, 1) ** 1.5 + 1.2 * tool * np.clip(lum - 0.15, 0, 1) + 0.02
    gq = gate[::4, ::4]
    p = (gq / gq.sum()).ravel()
    N = [25, 50, 90, 150, 240][s - 1]
    idx = rng.choice(len(p), size=N, p=p)
    L = np.zeros((h, w), np.float32)
    wq = gq.shape[1]
    for i in idx:
        y, x = int((i // wq) * 4 + rng.integers(0, 4)), int((i % wq) * 4 + rng.integers(0, 4))
        r = (0.8 + rng.uniform(0, 2.2)) * sc * (1 + 0.2 * s)
        ra = r * (1.0 if rng.random() < 0.6 else rng.uniform(1.5, 3.5))
        th = rng.uniform(0, np.pi)
        amp = rng.uniform(1.0, 1.8)
        R = int(3 * max(r, ra)) + 2
        ya, yb, xa, xb = max(0, y - R), min(h, y + R + 1), max(0, x - R), min(w, x + R + 1)
        if ya >= yb or xa >= xb:
            continue
        gy, gx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        dx, dy = gx - x, gy - y
        u = (dx * np.cos(th) + dy * np.sin(th)) / ra
        v = (-dx * np.sin(th) + dy * np.cos(th)) / r
        L[ya:yb, xa:xb] = np.maximum(L[ya:yb, xa:xb], np.clip(amp * np.exp(-2.0 * (u * u + v * v)), 0, 1))
    gain = [0.15, 0.20, 0.28, 0.35, 0.45][s - 1]
    B = 0.5 * cv2.GaussianBlur(L, (0, 0), 0.006 * w + 1) + 0.3 * cv2.GaussianBlur(L, (0, 0), 0.02 * w + 1)
    tint = np.array([1.0, 0.98, 0.95], np.float32)
    out = 1 - (1 - J) * (1 - L[..., None] * tint) * (1 - gain * np.clip(B * 3.0, 0, 1)[..., None] * tint)
    return _u8(out)


def occlusion(img, s, rng, return_mask=False):
    """Extra instrument shaft (shaded metal cylinder) crossing the view. Width is solved so the shaft covers
    5/10/20/30/40 percent of the frame."""
    h, w = img.shape[:2]
    f = [0.05, 0.10, 0.20, 0.30, 0.40][s - 1]
    sc = _scale(h, w)
    J = _f32(img)
    th = rng.uniform(0, np.pi)
    d = np.array([np.cos(th), np.sin(th)], np.float32)
    nrm = np.array([-d[1], d[0]], np.float32)
    # chord direction: avoid shafts almost parallel to a short border only when it would be pointless; any angle is fine
    p0 = np.array([rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h], np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    v = (xx - p0[0]) * nrm[0] + (yy - p0[1]) * nrm[1]
    u = (xx - p0[0]) * d[0] + (yy - p0[1]) * d[1]
    av = np.abs(v)
    target = f * h * w
    lo, hi = 0.0, float(np.hypot(h, w))
    sub = av[::2, ::2]
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if (sub <= mid).sum() * 4 < target:
            lo = mid
        else:
            hi = mid
    hw = 0.5 * (lo + hi)
    mask = np.clip((hw - av) / (1.2 * sc) + 0.5, 0, 1)
    t = np.clip(v / hw, -1, 1)
    body = np.sqrt(np.clip(1 - t * t, 0, 1))
    shade = 0.28 + 0.60 * np.clip(0.80 * body - 0.40 * t, 0, 1)
    spec = np.exp(-((t + 0.38) / 0.10) ** 2) * 0.75 + np.exp(-((t - 0.55) / 0.20) ** 2) * 0.18
    tex = 1 + 0.06 * (fractal_field(h, w, rng, octaves=3, base=2) - 0.5)
    ring = 1 - 0.12 * (np.cos(u / (0.07 * max(h, w)) + rng.uniform(0, 6)) > 0.92)
    metal = np.array([0.60, 0.62, 0.67], np.float32)
    shaft = np.clip(metal * (shade * tex * ring)[..., None] + (spec * 0.9)[..., None], 0, 1)
    shaft = shaft * (1 - 0.35 * (1 - body) ** 2)[..., None]  # dark rim
    sh = cv2.GaussianBlur(np.roll(mask, (int(6 * sc), int(8 * sc)), (0, 1)), (0, 0), 7 * sc + 1)
    J = J * (1 - 0.40 * (sh * (1 - mask)))[..., None]
    out = J * (1 - mask[..., None]) + shaft * mask[..., None]
    if return_mask:
        return mask
    return _u8(out)


KIND_FN = {"SMOKE": smoke, "BLOOD": blood, "FOG": fog, "GLARE": glare, "OCCLUSION": occlusion}


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)


def to_model_res(img):
    """Approximate model-input resolution (config image_target_size 236x414)."""
    return cv2.resize(img, tuple(CFG["model_hw"]), interpolation=cv2.INTER_AREA)


def psnr_model(a, b):
    return psnr(to_model_res(a), to_model_res(b))


def _bisect(fn, target, lo, hi, it=16):
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def match_controls(img, smoke_img, rng):
    """GAUSS and BRIGHTNESS images whose MSE to the clean view, measured AFTER resize to model-input resolution,
    equals SMOKE's. (Matching at native resolution lets the resize average the noise away, B5.)"""
    J = img.astype(np.float32)
    Jm = to_model_res(img).astype(np.float32)
    target = float(np.mean((to_model_res(smoke_img).astype(np.float32) - Jm) ** 2))
    z = rng.standard_normal(img.shape).astype(np.float32)

    def mse_after(arr):
        return float(np.mean((to_model_res(np.clip(np.rint(arr), 0, 255).astype(np.uint8)).astype(np.float32) - Jm) ** 2))

    sigma = _bisect(lambda sg: mse_after(J + sg * z), target, 0, 255, it=16)
    delta = _bisect(lambda d: mse_after(J + d), target, 0, 255, it=18)
    g = np.clip(np.rint(J + sigma * z), 0, 255).astype(np.uint8)
    b = np.clip(np.rint(J + delta), 0, 255).astype(np.uint8)
    return g, b


def corrupt_view(img, kind, s, seed, view=0):
    """Deterministic for (img 3-D uint8, kind, severity, seed, view). Returns (image, native-res PSNR vs clean)."""
    assert img.ndim == 3 and img.shape[-1] == 3, f"corrupt_view wants (H,W,3), got {img.shape}"
    if kind in KIND_FN:
        out = KIND_FN[kind](img, s, _rng(seed, view, kind))
    else:  # controls: calibrated against the SMOKE image of the same view/seed/severity
        sm = smoke(img, s, _rng(seed, view, "SMOKE"))
        g, b = match_controls(img, sm, _rng(seed, view, kind))
        out = g if kind == "GAUSS" else b
    return out, psnr(img, out)


def corrupt_views(views, kind, s, seed, which=None):
    """views: dict name -> (1,T,H,W,3) uint8 (the policy's observation layout).
    `which`: names of views to corrupt (None = all); others pass through unchanged.
    Returns (dict of (1,T,H,W,3), PSNR at model resolution of the first corrupted view)."""
    out, ps = {}, None
    for vi, (k, arr) in enumerate(views.items()):
        assert arr.ndim == 5 and arr.shape[0] == 1, f"expected (1,T,H,W,3), got {arr.shape}"
        if which is not None and k not in which:
            out[k] = arr
            continue
        frames = []
        for ti in range(arr.shape[1]):
            c, _ = corrupt_view(np.ascontiguousarray(arr[0, ti]), kind, s, seed, vi)
            frames.append(c)
            if ps is None:
                ps = psnr_model(arr[0, ti], c)
        out[k] = np.stack(frames)[None]
    return out, (float("nan") if ps is None else float(ps))


# ----------------------------------------------------------------------------- stage: data (kernel or venv)
def _hf():
    # <mock>
    if MOCK:
        return mock_gr00t.FakeApi(), mock_gr00t.fake_dl
    # </mock>
    from huggingface_hub import HfApi, hf_hub_download
    return HfApi(), hf_hub_download


def stage_data():
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    api, dl = _hf()
    repo = CFG["data_repo"]
    HZ = CFG["horizon"]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Discover real LeRobot roots (folders holding meta/info.json) by walking the tree.
    def ls(path):
        try:
            return list(api.list_repo_tree(repo, path_in_repo=path, repo_type="dataset", recursive=False))
        except Exception as e:
            log("list failed:", path, repr(e)[:120]); return []
    roots, frontier = [], [(p, 0) for p in ("Surgical", "cmr-surgical-60hz")]
    try:
        top = [x.path for x in api.list_repo_tree(repo, repo_type="dataset", recursive=False)]
        log("repo top level:", top[:30])
        frontier += [(p, 0) for p in top if p not in ("Surgical", "Endoscopy", "Ultrasound") and "." not in p]
    except Exception as e:
        log("top listing failed:", repr(e)[:120])
    seen = set()
    while frontier and len(roots) < 200:
        path, d = frontier.pop(0)
        if path in seen or d > 5: continue
        seen.add(path)
        kids = ls(path)
        names = {k.path.rsplit("/", 1)[-1]: k.path for k in kids}
        if "meta" in names:
            roots.append(path); continue
        frontier += [(k.path, d + 1) for k in kids if not hasattr(k, "size") or k.__class__.__name__ == "RepoFolder"]
    log(f"found {len(roots)} LeRobot roots, e.g.:", roots[:25])
    # Map the candidate order onto discovered roots by keyword.
    ordered = []
    for base, tag, folder in CANDIDATES:
        key = base.rsplit("/", 1)[-1].lower().split("_extracted")[0]
        for r in roots:
            if key in r.lower() and (r, tag, folder) not in ordered:
                ordered.append((r, tag, folder))
    log("candidate roots in priority order:", [o[0] for o in ordered][:10])
    chosen = None
    for base, tag, folder in ordered:
        try:
            p = dl(repo, f"{base}/meta/info.json", repo_type="dataset", local_dir=DATA_DIR)
        except Exception as e:
            log(f"candidate {base}: no meta/info.json ({type(e).__name__})")
            continue
        info = jload(p)
        if int(info.get("total_episodes", 0)) < CFG["n_episodes"]:
            log(f"candidate {base}: only {info.get('total_episodes')} episodes, skip")
            continue
        chosen = (base, tag, folder, info)
        break
    if chosen is None:
        raise RuntimeError("No candidate dataset reachable.")
    base, tag, folder, info = chosen
    log(f"CHOSEN dataset: {base}  tag={tag}  episodes={info['total_episodes']}  fps={info.get('fps')}")
    if "nephfat" in base.lower():
        log("NOTE: NephFat is in the GR00T-H-N1.7 training mix (open_h/gr00t_h_config.yaml): these are TRAINING frames.")

    root = DATA_DIR / base
    meta_files = [x.path for x in api.list_repo_tree(repo, path_in_repo=f"{base}/meta", repo_type="dataset", recursive=False)
                  if getattr(x, "size", None) is not None or x.path.endswith((".json", ".jsonl"))]
    for mf in meta_files:
        try:
            dl(repo, mf, repo_type="dataset", local_dir=DATA_DIR)
        except Exception as e:
            log("meta file skipped:", mf, repr(e)[:100])
    mj = root / "meta" / "modality.json"
    if not mj.exists():
        mj.write_bytes(urllib.request.urlopen(RAW_GH.format(folder=folder), timeout=60).read())
        log("modality.json fetched from GR00T-H repo ->", mj)
    modality = jload(mj)
    vkeys = [v["original_key"] for v in modality["video"].values()]
    log("video keys:", vkeys)

    # position in episodes.jsonl is what LeRobotEpisodeLoader indexes by; episode_index names the files
    eps = []
    for pos, line in enumerate([l for l in (root / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]):
        d = json.loads(line)
        if d.get("length", 0) >= HZ + 70:
            eps.append((int(d["episode_index"]), pos, int(d["length"])))
    eps.sort()
    log(f"{len(eps)} usable episodes (length>={HZ + 70})")
    n = CFG["n_episodes"]
    order = list(np.linspace(0, len(eps) - 1, n).round().astype(int))  # evenly spaced
    spare = [i for i in range(len(eps)) if i not in set(order)]
    cs = info.get("chunks_size", 1000)

    def fetch(i):
        e, pos, L = eps[i]
        ch = e // cs
        rels = [info["data_path"].format(episode_chunk=ch, episode_index=e)]
        rels += [info["video_path"].format(episode_chunk=ch, episode_index=e, video_key=k) for k in vkeys]
        for r in rels:
            dl(repo, f"{base}/{r}", repo_type="dataset", local_dir=DATA_DIR)
        return i

    good, tried = [], set()
    queue = list(order)
    while len(good) < n and queue:
        batch = [i for i in queue[: n - len(good)] if i not in tried]
        queue = queue[len(batch):]
        tried.update(batch)
        with ThreadPoolExecutor(8) as ex:
            futs = {i: ex.submit(fetch, i) for i in batch}
        for i, f in futs.items():
            try:
                f.result()
                good.append(i)
            except Exception as e:
                log(f"episode {eps[i][0]} download failed ({type(e).__name__}); will substitute a spare")
                if spare:
                    queue.append(spare.pop(len(spare) // 2))
    good = sorted(good)
    if len(good) < n:
        log(f"WARNING only {len(good)} episodes obtained")
    episodes = [dict(ep=eps[i][0], pos=eps[i][1], length=eps[i][2]) for i in good]
    frames, fid = [], 0
    for rank, d in enumerate(episodes):
        ts = np.unique(np.linspace(0, d["length"] - HZ - 1, CFG["t_per_ep"]).round().astype(int))
        for j, t in enumerate(ts):
            frames.append(dict(fid=fid, ep=d["ep"], pos=d["pos"], rank=rank, j=j, t=int(t), L=d["length"]))
            fid += 1
    sel = dict(base=base, tag=tag, folder=folder, root=str(root), episodes=episodes, frames=frames,
               video_keys=vkeys, fps=info.get("fps"), cfg_hash=CFG_HASH, hero_rank=len(episodes) // 2)
    jdump(sel, OUT / "selection.json")
    log(f"selection saved: {len(episodes)} episodes, {len(frames)} frames")
    log("dataset:", "primary JHU dVRK" if "JHU" in base else "fallback: " + base)


# ----------------------------------------------------------------------------- GPU helpers
def _pixel_hash(batch):
    """Short hash of the first image tensor found in a collated model input (shows that corrupted inputs reach the model)."""
    stack = [batch]
    while stack:
        x = stack.pop()
        if hasattr(x, "items"):
            for k, v in x.items():
                if "pixel" in str(k) and hasattr(v, "detach"):
                    return hashlib.md5(v.detach().float().cpu().numpy().tobytes()).hexdigest()[:10]
                stack.append(v)
    return None


def log_checkpoint_config(pol):
    """Log the loaded checkpoint's state-dropout setting for the embodiment and the processor's image sizes."""
    try:
        from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import EMBODIMENT_TAG_TO_PROJECTOR_INDEX
        idx = EMBODIMENT_TAG_TO_PROJECTOR_INDEX[pol.embodiment_tag.value]
        probs = pol.model.action_head.dropout_prob_by_embodiment
        log(f"state dropout probability for {pol.embodiment_tag.value} (projector index {idx}): {float(probs[idx]):.3f}")
    except Exception as e:
        log("state dropout configuration not readable:", repr(e)[:160])
    try:
        pr = pol.processor
        log("processor image sizes:", {a: getattr(pr, a, None) for a in ("image_target_size", "image_crop_size")})
    except Exception as e:
        log("processor image sizes not readable:", repr(e)[:160])


def make_policy(model_id=None):
    """Gr00tPolicy subclass: repo hard-codes bfloat16 (no HW support on T4), so dtype is switchable.
    Also keeps the normalized action prediction (needed for the gripper sign test)."""
    import torch
    # <mock>
    if MOCK:
        return mock_gr00t.make_mock_policy(jload(OUT / "selection.json")["tag"])
    # </mock>
    from gr00t.data.types import MessageType
    from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

    class DTypePolicy(Gr00tPolicy):
        last_norm = None
        last_pixel_hash = None
        zero_state_input = False   # ablation: zero the proprioceptive state fed to the model (decoding keeps the true reference state)

        def set_dtype(self, dt):
            self.model.to(dtype=dt)
            self.dtype = dt
            torch.cuda.empty_cache()

        def _get_action(self, observation, options=None):  # mirrors Gr00tPolicy._get_action with variable dtype
            unb = self._unbatch_observation(observation)
            processed, states = [], []
            for obs in unb:
                v = self._to_vla_step_data(obs)
                states.append(v.states)
                if self.zero_state_input:
                    v = copy.copy(v)
                    v.states = {k: np.zeros_like(a) for k, a in v.states.items()}
                processed.append(self.processor([{"type": MessageType.EPISODE_STEP.value, "content": v}]))
            coll = _rec_to_dtype(self.collate_fn(processed), dtype=self.dtype)
            self.last_pixel_hash = _pixel_hash(coll)
            with torch.inference_mode():
                pred = self.model.get_action(**coll)
            na = pred["action_pred"].float()
            self.last_norm = na.cpu().numpy()
            bs = {k: np.stack([s[k] for s in states], 0) for k in self.modality_configs["state"].modality_keys}
            ua = self.processor.decode_action(self.last_norm, self.embodiment_tag, bs)
            return {k: v.astype(np.float32) for k, v in ua.items()}, {}

    sel = jload(OUT / "selection.json")
    from huggingface_hub import snapshot_download
    path = snapshot_download(model_id or CFG["model_id"])
    pol = DTypePolicy(embodiment_tag=sel["tag"], model_path=path, device="cuda:0", strict=True)
    pol.set_dtype(torch.float16)
    log_checkpoint_config(pol)
    return pol


def attn_report(pol):
    import torch
    try:
        import flash_attn  # noqa
        log("WARNING flash_attn is installed; T4 cannot run FA2. Uninstall it.")
    except Exception:
        log("flash_attn not installed -> sdpa path expected")
    seen = set()
    for n, m in pol.model.named_modules():
        c = getattr(m, "config", None)
        a = getattr(c, "_attn_implementation", None)
        if a and a not in seen:
            seen.add(a)
            log(f"attn implementation seen at {n or 'root'}: {a}")
    log("model params dtype:", next(pol.model.parameters()).dtype, "| cuda mem GB:", round(torch.cuda.memory_allocated() / 1e9, 2))


def _loader_cls():
    # <mock>
    if MOCK:
        return mock_gr00t.MockLoader
    # </mock>
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    return LeRobotEpisodeLoader


def _extract_step_data():
    # <mock>
    if MOCK:
        return mock_gr00t.extract_step_data
    # </mock>
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    return extract_step_data


def make_loader(pol, sel):
    """LeRobotEpisodeLoader without stats.json (unused here), first backend that decodes one sparse frame."""
    LeRobotEpisodeLoader = _loader_cls()
    errs = []
    for be in CFG["video_backends"]:
        try:
            ld = LeRobotEpisodeLoader(dataset_path=sel["root"], modality_configs=pol.get_modality_config(),
                                      video_backend=be, video_backend_kwargs=None, require_stats=False)
            d0 = sel["episodes"][0]
            df = ld.get_sparse_episode(d0["pos"], np.clip(np.arange(0, CFG["horizon"]), 0, d0["length"] - 1), video_indices=[0], allow_padding=True)
            ld.backend_name = be
            log(f"video backend OK: {be}; sparse rows={len(df)}")
            return ld
        except Exception as e:
            errs.append(f"{be}: {e!r}")
            log(f"video backend {be} failed: {e!r}"[:600])
    raise RuntimeError("no video backend works:\n" + "\n".join(errs))


class Roles:
    """Splits action keys into pose (xyz first) and gripper keys; builds obs and GT from sparse trajectories."""

    def __init__(self, pol, loader, sel):
        from copy import deepcopy
        mc = pol.get_modality_config()
        keys = list(mc["action"].modality_keys)
        self.keys = keys
        self.pose = [k for k in keys if "pose" in k or "eef" in k]
        self.grip = [k for k in keys if "grip" in k or "jaw" in k]
        self.vkeys = list(mc["video"].modality_keys)
        self.endo = next((k for k in self.vkeys if "endo" in k.lower()), self.vkeys[0])
        self.loader, self.tag = loader, pol.embodiment_tag
        self.bgr = getattr(loader, "backend_name", "") == "opencv"
        self.mc_obs = deepcopy(loader.modality_configs)
        self.mc_obs.pop("action")
        # normalized action layout: pose keys 9D (xyz + rot6d), other keys 1D, in modality-key order
        off, self.grip_idx = 0, []
        for k in keys:
            d = 9 if k in self.pose else 1
            if k in self.grip:
                self.grip_idx.append(off)
            off += d
        self.norm_dim = off
        self.H = None
        log("pose keys:", self.pose, "| gripper keys:", self.grip, "| views:", self.vkeys, "| corrupted (primary):", self.endo,
            "| normalized gripper idx:", self.grip_idx)

    def traj(self, d, t):
        """Sparse episode rows t..t+H-1 (clamped) with only frame t decoded."""
        rows = np.clip(np.arange(t, t + CFG["horizon"]), 0, d["length"] - 1)
        return self.loader.get_sparse_episode(d["pos"], rows, video_indices=[t], allow_padding=True)

    def obs(self, traj, t):
        dp = _extract_step_data()(traj, t, self.mc_obs, self.tag)
        video = {k: np.array(v)[None] for k, v in dp.images.items()}
        if self.bgr:
            video = {k: np.ascontiguousarray(v[..., ::-1]) for k, v in video.items()}
        state = {k: np.asarray(v, np.float32)[None] for k, v in dp.states.items()}
        lang = {k: [[dp.text]] for k in self.loader.modality_configs["language"].modality_keys}
        return {"video": video, "state": state, "language": lang}

    def gt(self, traj, t, H):
        hi = int(traj.index.max())
        rows = np.minimum(np.arange(t, t + H), hi)
        xyz = np.stack([np.stack(traj.loc[rows, f"action.{k}"].values)[:, :3] for k in self.pose], 1)  # (H,A,3)
        gr = (np.stack([np.stack(traj.loc[rows, f"action.{k}"].values).reshape(H, -1)[:, 0] for k in self.grip], 1)
              if self.grip else np.zeros((H, 0)))
        return xyz, gr

    def split(self, pol, action):
        Hs = min(CFG["horizon"], action[self.pose[0]].shape[1])
        self.H = Hs
        xyz = np.stack([action[k][0, :Hs, :3] for k in self.pose], 1)  # (H,A,3)
        gr = (np.stack([action[k][0, :Hs, 0] for k in self.grip], 1) if self.grip else np.zeros((Hs, 0)))
        ln = pol.last_norm
        if ln is not None and ln.shape[-1] >= self.norm_dim and self.grip_idx:
            gn = ln[0, :Hs][:, self.grip_idx]
        else:
            gn = np.full_like(gr, np.nan)
        return xyz, gr, gn


def infer(pol, roles, obs, seed, raise_oom=False):
    """Returns (xyz (H,A,3), decoded gripper (H,G), normalized gripper (H,G)). NaN-filled on persistent OOM unless raise_oom."""
    import torch
    for attempt in range(2):
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed % (2**32 - 1))
            act, _ = pol.get_action(obs)
            return roles.split(pol, act)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if raise_oom:
                raise
    H = roles.H or CFG["horizon"]
    return (np.full((H, len(roles.pose), 3), np.nan), np.full((H, len(roles.grip)), np.nan), np.full((H, len(roles.grip)), np.nan))


def with_views(obs, views):
    o = dict(obs)
    o["video"] = views
    return o


def dtype_of(name):
    import torch
    return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def win_mean_dist(a, b, n):
    """mean over first n steps of per-step Euclidean distance over arms and xyz. a,b (H,A,3)."""
    return float(np.sqrt(((a[:n] - b[:n]) ** 2).sum(axis=(-1, -2))).mean())


# ----------------------------------------------------------------------------- stage: model_check (GPU)
def stage_model_check():
    import torch
    sel = jload(OUT / "selection.json")
    W8 = CFG["win_short"]
    pol = make_policy()
    attn_report(pol)
    loader = make_loader(pol, sel)
    roles = Roles(pol, loader, sel)
    step = max(1, len(sel["frames"]) // CFG["fp_check_frames"])
    frames = sel["frames"][::step][: CFG["fp_check_frames"]]
    eps = {e["ep"]: e for e in sel["episodes"]}

    def run(dtype):
        pol.set_dtype(dtype)
        res = []
        for f in frames:
            tr = roles.traj(eps[f["ep"]], f["t"])
            obs = roles.obs(tr, f["t"])
            xyz, _, _ = infer(pol, roles, obs, frame_seed(f["fid"]), raise_oom=True)
            gx, _ = roles.gt(tr, f["t"], len(xyz))
            res.append((xyz, gx))
        return res

    r16 = run(torch.float16)
    allx = np.concatenate([g.reshape(-1, 3) for _, g in r16])
    log(f"GT xyz native units: median |x| {np.nanmedian(np.abs(allx)):.4f}, range [{np.nanmin(allx):.3f}, {np.nanmax(allx):.3f}]")
    scale = 1000.0 if np.nanmedian(np.abs(allx)) < 5 else 1.0
    ref_name, ref = None, None
    for name, dt in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        try:
            cand = run(dt)
        except Exception as e:  # includes OOM (raise_oom=True)
            log(f"{name} reference failed: {e!r}"[:300])
            torch.cuda.empty_cache()
            continue
        if np.isnan(np.array([a for a, _ in cand])).any():
            log(f"{name} reference contains NaN -> unusable")
            continue
        ref_name, ref = name, cand
        break
    if ref is None:
        raise RuntimeError("no valid fp32/bf16 reference: cannot validate fp16. Technical failure.")
    diffs = [win_mean_dist(a, b, W8) * scale for (a, _), (b, _) in zip(r16, ref)]
    errs = [win_mean_dist(b, g, W8) * scale for (b, g) in ref]
    mean_err = float(np.mean(errs))
    rel = float(np.max(diffs)) / max(mean_err, 1e-9)
    nan16 = bool(np.isnan(np.array([a for a, _ in r16])).any())
    passed = (rel < CFG["fp_check_tol"]) and not nan16
    out = dict(reference=ref_name, max_diff_mm=float(np.max(diffs)), mean_clean_vs_gt_mm=mean_err, rel=rel,
               tol=CFG["fp_check_tol"], nan_fp16=nan16, passed=passed, unit_scale=scale, window=W8,
               final_dtype="float16" if passed else ref_name,
               frames_per_ep=CFG["t_per_ep"] if passed else CFG["fp32_frames_per_ep"])
    jdump(out, OUT / "fp_check.json")
    log("FP CHECK", json.dumps(out))
    if not passed:
        log(f"fp16 check FAILED -> main run uses {ref_name} on {out['frames_per_ep']} frames/episode")


# ----------------------------------------------------------------------------- stage: diag (GPU)
def infer_full(pol, roles, obs, seed, zero_state=False):
    """One forward pass at a fixed seed; returns the decoded action dict {key: (1, H, D)} (absolute poses, native units).

    A persistent CUDA out-of-memory error yields a NaN-filled dict so the frame is reported, not lost.
    """
    import torch
    for attempt in range(2):
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed % (2**32 - 1))
            pol.zero_state_input = zero_state
            act, _ = pol.get_action(obs)
            return act
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
        finally:
            pol.zero_state_input = False
    H = roles.H or CFG["horizon"]
    return {k: np.full((1, H, 7 if k in roles.pose else 1), np.nan, np.float32) for k in roles.keys}


def blank_views(views, which=None):
    """Replace views (all, or the keys in `which`) by a constant gray at each image's mean intensity."""
    out = {}
    for k, v in views.items():
        out[k] = np.full_like(v, int(round(float(v.mean())))) if (which is None or k in which) else v
    return out


def concat_action(act, keys, n):
    """First n steps of every action key, concatenated along the last axis (layout of the standalone inference script)."""
    return np.concatenate([np.atleast_1d(act[k][0, :n]).reshape(n, -1) for k in keys], axis=-1)


def gt_concat(traj, t, keys, n):
    rows = np.minimum(np.arange(t, t + n), int(traj.index.max()))
    return np.concatenate([np.stack(traj.loc[rows, f"action.{k}"].values).reshape(n, -1) for k in keys], axis=-1)


def run_key():
    """Identifies a run by configuration and selected frames; checkpoints written under another key are not reused."""
    sel_hash = hashlib.sha256((OUT / "selection.json").read_bytes()).hexdigest()[:12]
    return f"{CFG_HASH}:{sel_hash}"


def load_checkpoint(path, key):
    """Rows of a jsonl checkpoint written under `key`. A file containing rows of another key is moved aside."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    if any(r.get("key") != key for r in rows):
        aside = path.with_name(f"{path.stem}.stale{int(time.time())}{path.suffix}")
        path.rename(aside)
        log(f"checkpoint {path.name} belongs to a different run; moved to {aside.name}")
        return []
    return rows


DIAG_CONDS = ("clean", "clean2", "ens2", "ens3", "clean_rep", "blank_endo", "blank_all", "state_zero")


def stage_diag():
    """Alignment and modality-use diagnostics on an evenly spaced subset of the selected frames (no corruption sweep).

    Rows are appended per frame; a restarted run continues where the checkpoint ends.
    """
    sel = jload(OUT / "selection.json")
    fpc = jload(OUT / "fp_check.json")
    sc, AH = fpc["unit_scale"], CFG["win_short"]
    pol = make_policy()
    pol.set_dtype(dtype_of(fpc["final_dtype"]))
    loader = make_loader(pol, sel)
    roles = Roles(pol, loader, sel)
    eps = {e["ep"]: e for e in sel["episodes"]}
    keep_j = None if fpc["frames_per_ep"] >= CFG["t_per_ep"] else set(range(0, CFG["t_per_ep"], 2))
    pool = [f for f in sel["frames"] if keep_j is None or f["j"] in keep_j]
    take = np.unique(np.linspace(0, len(pool) - 1, min(CFG["diag_frames"], len(pool))).round().astype(int))
    frames = [pool[i] for i in take]
    path, key = OUT / "diag.jsonl", run_key()
    rows = load_checkpoint(path, key)
    done = {r["fid"] for r in rows}
    t0, n_new = time.time(), 0
    with open(path, "a") as fh:
        for f in frames:
            if f["fid"] in done:
                continue
            tr = roles.traj(eps[f["ep"]], f["t"])
            obs = roles.obs(tr, f["t"])
            seed = frame_seed(f["fid"])
            views = obs["video"]
            plan = {"clean": (obs, seed, False), "clean2": (obs, seed + 1, False), "ens2": (obs, seed + 2, False), "ens3": (obs, seed + 3, False),
                    "clean_rep": (obs, seed, False),
                    "blank_endo": (with_views(obs, blank_views(views, [roles.endo])), seed, False),
                    "blank_all": (with_views(obs, blank_views(views)), seed, False),
                    "state_zero": (obs, seed, True)}
            acts, px = {}, {}
            for c, (o, sd, zs) in plan.items():
                acts[c] = infer_full(pol, roles, o, sd, zero_state=zs)
                px[c] = getattr(pol, "last_pixel_hash", None)
            xyz = {c: roles.split(pol, a)[0] * sc for c, a in acts.items()}
            H = xyz["clean"].shape[0]
            gx = roles.gt(tr, f["t"], H)[0] * sc
            sx = np.stack([obs["state"][k][0, 0, :3] for k in roles.pose]) * sc
            nv = dict(mse=np.nan, mae=np.nan, zero_mse=np.nan, zero_mae=np.nan, dims=0)
            if not np.isnan(xyz["clean"]).any():
                # reference-script metric: MSE/MAE over the concatenated native-unit action vector, first AH steps
                pred16, gt16 = concat_action(acts["clean"], roles.keys, AH), gt_concat(tr, f["t"], roles.keys, AH)
                nv.update(mse=float(np.mean((pred16 - gt16) ** 2)), mae=float(np.mean(np.abs(pred16 - gt16))), dims=int(gt16.shape[1]))
                try:
                    hold16 = np.tile(np.concatenate([obs["state"][k][0, 0] for k in roles.keys])[None], (AH, 1))
                    assert hold16.shape == gt16.shape
                    nv.update(zero_mse=float(np.mean((hold16 - gt16) ** 2)), zero_mae=float(np.mean(np.abs(hold16 - gt16))))
                except (KeyError, AssertionError):
                    pass
            fh.write(json.dumps(dict(key=key, fid=f["fid"], ep=f["ep"], t=f["t"], H=H, state=sx.tolist(), gt=gx.tolist(), px=px,
                                     pred={c: v.tolist() for c, v in xyz.items()}, nv=nv)) + "\n")
            fh.flush(); os.fsync(fh.fileno())
            n_new += 1
            if n_new % 5 == 0:
                log(f"diag {len(done) + n_new}/{len(frames)}  {(time.time() - t0) / n_new:.1f}s/frame")
    rows = load_checkpoint(path, key)
    out = diag_report(rows, sc)
    jdump({k: v for k, v in out.items() if k != "block"}, OUT / "diag.json")


def diag_report(rows, unit_scale=1000.0, verbose=True):
    """Aggregate diagnostics, print the DIAG block, return a dict (no gating: SANITY is informational).

    Zero-motion baseline = hold the current end-effector position (relative action 0), which uses no ground truth.
    The oracle variant holds GT step 0 and is reported for scale only. SANITY compares the 4-seed ensemble clean error
    with the hold-state error at the full horizon; NaN yields N/A.
    """
    n8 = CFG["win_short"]
    H = rows[0]["H"]
    wH = str(H)
    X = {c: np.array([r["pred"][c] for r in rows]) for c in DIAG_CONDS}      # (F,H,A,3) mm
    GT = np.array([r["gt"] for r in rows])                                     # (F,H,A,3)
    ST = np.array([r["state"] for r in rows])                                  # (F,A,3)
    ens = np.mean([X[c] for c in ("clean", "clean2", "ens2", "ens3")], axis=0)
    hold = np.repeat(ST[:, None], H, axis=1)
    oracle = np.repeat(GT[:, :1], H, axis=1)

    def d(a, b, n):  # mean over first n steps of the per-step Euclidean distance over both arms -> (F,)
        return np.sqrt(((a[:, :n] - b[:, :n]) ** 2).sum(axis=(-1, -2))).mean(-1)

    M = {}
    for n, w in ((n8, "8"), (H, wH)):
        M[f"zero_{w}"], M[f"oracle_{w}"] = d(hold, GT, n), d(oracle, GT, n)
        M[f"model_{w}"], M[f"ens_{w}"] = d(X["clean"], GT, n), d(ens, GT, n)
        M[f"floor_{w}"], M[f"rep_{w}"] = d(X["clean2"], X["clean"], n), d(X["clean_rep"], X["clean"], n)
        for c in ("blank_endo", "blank_all", "state_zero"):
            M[f"drift_{c}_{w}"] = d(X[c], X["clean"], n)
            M[f"err_{c}_{w}"] = d(X[c], GT, n)
    corr = {}
    for n, w in ((n8, "8"), (H, wH)):
        pd_ = (X["clean"][:, :n] - ST[:, None]).reshape(-1, 3)
        gd_ = (GT[:, :n] - ST[:, None]).reshape(-1, 3)
        ok = ~(np.isnan(pd_).any(1) | np.isnan(gd_).any(1))
        corr[w] = []
        for a in range(3):
            p, g = pd_[ok, a], gd_[ok, a]
            good = len(p) > 2 and p.std() > 0 and g.std() > 0
            corr[w].append(dict(axis="xyz"[a], r=float(np.corrcoef(p, g)[0, 1]) if good else float("nan"),
                                slope=float(np.polyfit(g, p, 1)[0]) if good else float("nan")))
    col = lambda k: np.array([r["nv"][k] for r in rows], float)
    nv = dict(mse=float(np.nanmean(col("mse"))) if np.isfinite(col("mse")).any() else float("nan"),
              mae=float(np.nanmean(col("mae"))) if np.isfinite(col("mae")).any() else float("nan"),
              zero_mse=float(np.nanmean(col("zero_mse"))) if np.isfinite(col("zero_mse")).any() else float("nan"),
              zero_mae=float(np.nanmean(col("zero_mae"))) if np.isfinite(col("zero_mae")).any() else float("nan"),
              dims=int(max(r["nv"]["dims"] for r in rows)))
    mean = lambda k: float(np.nanmean(M[k])) if np.isfinite(M[k]).any() else float("nan")
    n_nan = int(np.isnan(M[f"model_{wH}"]).sum())
    ens_e, zero_e = mean(f"ens_{wH}"), mean(f"zero_{wH}")
    verdict = "N/A" if (np.isnan(ens_e) or np.isnan(zero_e)) else ("PASS" if ens_e < zero_e else "FAIL")
    pxs = [r.get("px") or {} for r in rows]
    px_ok = [(p.get("clean") is not None) for p in pxs]
    px_diff = {c: int(sum(p.get(c) != p.get("clean") for p, ok in zip(pxs, px_ok) if ok)) for c in ("blank_endo", "blank_all", "clean_rep")}
    out = dict(n_frames=len(rows), n_nan_frames=n_nan, horizon=H, unit_scale=unit_scale, means={k: mean(k) for k in M}, corr=corr, nvidia=nv,
               sanity=verdict, sanity_oracle_pass=bool(mean(f"ens_{wH}") < mean(f"oracle_{wH}")), pixel_hash_differs=px_diff,
               blank_endo_over_floor={w: mean(f"drift_blank_endo_{w}") / mean(f"floor_{w}") for w in ("8", wH)})
    L = ["", "=" * 78, "DIAG", "=" * 78,
         f"{len(rows)} frames ({n_nan} with NaN clean prediction) | positions in mm (native x {unit_scale:g}) | windows: first {n8} and all {H} steps | both arms"]
    if len(rows) <= 20:
        L.append("per frame (mm): fid zero8 model8 | zero%s model%s oracle%s | floor8 floor%s | drift8 blank_endo/blank_all/state_zero | drift%s same" % ((wH,) * 5))
        for i, r in enumerate(rows):
            L.append(f"  {r['fid']:>5} {M['zero_8'][i]:6.2f} {M['model_8'][i]:6.2f} | {M[f'zero_{wH}'][i]:6.2f} {M[f'model_{wH}'][i]:6.2f} {M[f'oracle_{wH}'][i]:6.2f}"
                     f" | {M['floor_8'][i]:5.2f} {M[f'floor_{wH}'][i]:5.2f} | " + "/".join(f"{M[f'drift_{c}_8'][i]:5.2f}" for c in ("blank_endo", "blank_all", "state_zero"))
                     + " | " + "/".join(f"{M[f'drift_{c}_{wH}'][i]:5.2f}" for c in ("blank_endo", "blank_all", "state_zero")))
    L += ["", f"{'mean over frames (mm)':<42}{'8 steps':>10}{wH + ' steps':>11}"]
    rowfmt = lambda name, k: f"{name:<42}{mean(k + '_8'):>10.3f}{mean(k + '_' + wH):>11.3f}"
    L += [rowfmt("zero-motion (hold state)", "zero"), rowfmt("zero-motion (oracle, hold GT step 0)", "oracle"), rowfmt("model clean (single seed)", "model"),
          rowfmt("model clean (mean of 4 seeds)", "ens"), rowfmt("seed-to-seed noise floor", "floor"), rowfmt("same seed rerun (expected 0)", "rep")]
    for c, nm in (("blank_endo", "blank endoscope"), ("blank_all", "all cameras blank"), ("state_zero", "state zeroed (model input only)")):
        L.append(rowfmt("drift vs clean: " + nm, "drift_" + c))
    for c, nm in (("blank_endo", "blank endoscope"), ("blank_all", "all cameras blank"), ("state_zero", "state zeroed (model input only)")):
        L.append(rowfmt("error vs GT: " + nm, "err_" + c))
    L += ["", f"drift(blank endoscope) / noise floor: {out['blank_endo_over_floor']['8']:.2f} (8 steps), {out['blank_endo_over_floor'][wH]:.2f} ({wH} steps)",
          "processor input differs from clean in (frames): " + ", ".join(f"{c} {v}/{sum(px_ok)}" for c, v in px_diff.items()) if any(px_ok) else "processor input hash unavailable",
          f"state-zeroed drift at {wH} steps: {mean(f'drift_state_zero_{wH}'):.4f} mm (about 0 expected when state dropout is 1.0)",
          "", f"reference-script metric ({nv['dims']}-D native-unit action, first {n8} steps, mean over frames):",
          f"  model  MSE {nv['mse']:.3e}  MAE {nv['mae']:.3e}", f"  hold-state baseline  MSE {nv['zero_mse']:.3e}  MAE {nv['zero_mae']:.3e}",
          "", "displacement-from-state correlation (pred vs GT), Pearson r | slope:"]
    for w in corr:
        L.append(f"  {w:>3} steps: " + "  ".join(f"{c['axis']} r={c['r']:+.2f} s={c['slope']:+.2f}" for c in corr[w]))
    L += ["", f"SANITY {verdict}: 4-seed ensemble clean error {ens_e:.3f} mm vs hold-state error {zero_e:.3f} mm at {H} steps (diagnostic only)",
          f"  oracle baseline: ensemble {'below' if out['sanity_oracle_pass'] else 'not below'} oracle zero-motion ({mean(f'oracle_{wH}'):.3f} mm)", "=" * 78]
    out["block"] = "\n".join(L)
    if verbose:
        print(out["block"])
    return out


# ----------------------------------------------------------------------------- stage: visual (GPU env for decoding)
def stage_visual():
    sel = jload(OUT / "selection.json")
    pol = make_policy()
    loader = make_loader(pol, sel)
    roles = Roles(pol, loader, sel)
    d = sel["episodes"][sel["hero_rank"]]
    hf = [f for f in sel["frames"] if f["rank"] == sel["hero_rank"]]
    t = hf[min(4, len(hf) - 1)]["t"]   # 5th frame of the hero episode (full run); last one in QUICK (3 frames/episode)
    obs = roles.obs(roles.traj(d, t), t)
    np.savez_compressed(OUT / "visual_frame.npz", **{k: v[0, 0] for k, v in obs["video"].items()})
    log("saved visual_frame.npz", {k: v.shape for k, v in obs["video"].items()})


# ----------------------------------------------------------------------------- stage: main (GPU)
def cond_names():
    prim = [f"{k}|{s}" for k in CFG["kinds"] for s in CFG["severities"]]
    sec = [f"{k}|{s}|all" for k in CFG["kinds"] for s in CFG["secondary_sevs"]]
    extra = ["clean3", "clean4", "blank_endo"] + [f"{k}|{CFG['ensemble_sev']}|s{j}" for k in CFG["artifacts"] for j in (1, 2, 3)]
    return ["clean", "clean2"] + prim + sec + extra


def stage_main():
    import torch
    sel = jload(OUT / "selection.json")
    fpc = jload(OUT / "fp_check.json")
    pol = make_policy()
    pol.set_dtype(dtype_of(fpc["final_dtype"]))
    attn_report(pol)
    loader = make_loader(pol, sel)
    roles = Roles(pol, loader, sel)
    eps = {e["ep"]: e for e in sel["episodes"]}
    conds = cond_names()
    res_path = OUT / "results.jsonl"
    (OUT / "frames").mkdir(exist_ok=True)
    key = run_key()
    done = {r["fid"] for r in load_checkpoint(res_path, key)}
    if done:
        log(f"resuming: {len(done)} frames already done")
    keep_j = None if fpc["frames_per_ep"] >= CFG["t_per_ep"] else set(range(0, CFG["t_per_ep"], 2))
    frames = [f for f in sel["frames"] if keep_j is None or f["j"] in keep_j]
    t0, n_new, nan_clean = time.time(), 0, 0
    fh = open(res_path, "a")
    meta = None
    for f in frames:
        if f["fid"] in done:
            continue
        d = eps[f["ep"]]
        tr = roles.traj(d, f["t"])
        obs = roles.obs(tr, f["t"])
        seed = frame_seed(f["fid"])
        pred, ps = {}, {}
        pred["clean"] = infer(pol, roles, obs, seed)
        if n_new == 0 and np.isnan(pred["clean"][0]).any():
            raise RuntimeError("frame 0 clean prediction is NaN: aborting (dtype/OOM problem)")
        pred["clean2"] = infer(pol, roles, obs, seed + 1)
        pred["clean3"] = infer(pol, roles, obs, seed + 2)
        pred["clean4"] = infer(pol, roles, obs, seed + 3)
        pred["blank_endo"] = infer(pol, roles, with_views(obs, blank_views(obs["video"], [roles.endo])), seed)
        H = pred["clean"][0].shape[0]
        gx, gg = roles.gt(tr, f["t"], H)
        for k in CFG["kinds"]:
            for s in CFG["severities"]:
                cv, p = corrupt_views(obs["video"], k, s, seed, which=[roles.endo])
                if s >= 3:
                    assert not np.array_equal(cv[roles.endo], obs["video"][roles.endo]), f"{k} severity {s} left the view unchanged"
                pred[f"{k}|{s}"] = infer(pol, roles, with_views(obs, cv), seed)
                if k in CFG["artifacts"] and s == CFG["ensemble_sev"]:
                    for j in (1, 2, 3):
                        pred[f"{k}|{s}|s{j}"] = infer(pol, roles, with_views(obs, cv), seed + j)
                ps[f"{k}|{s}"] = round(p, 3)
            for s in CFG["secondary_sevs"]:
                cv, p = corrupt_views(obs["video"], k, s, seed, which=None)
                pred[f"{k}|{s}|all"] = infer(pol, roles, with_views(obs, cv), seed)
        img = obs["video"][roles.endo][0, 0]
        cv2.imwrite(str(OUT / "frames" / f"{f['fid']}.jpg"), np.ascontiguousarray(img[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 96])
        if meta is None:
            meta = dict(cfg_hash=CFG_HASH, dtype=fpc["final_dtype"], unit_scale=fpc["unit_scale"], conds=conds, n_frames=len(frames),
                        pose_keys=roles.pose, grip_keys=roles.grip, views=roles.vkeys, corrupted_view=roles.endo, tag=sel["tag"],
                        base=sel["base"], H=int(H), win_short=CFG["win_short"])
            jdump(meta, OUT / "run_meta.json")
        sxyz = np.stack([obs["state"][k][0, 0, :3] for k in roles.pose]) if roles.pose else np.zeros((0, 3))
        line = dict(key=key, fid=f["fid"], ep=f["ep"], t=f["t"], gt_xyz=np.round(gx, 5).ravel().tolist(), gt_grip=np.round(gg, 4).ravel().tolist(),
                    state_xyz=np.round(sxyz, 5).ravel().tolist(), psnr=ps,
                    pred={c: dict(xyz=np.round(v[0], 5).ravel().tolist(), grip=np.round(v[1], 4).ravel().tolist(),
                                  gripn=np.round(v[2], 4).ravel().tolist()) for c, v in pred.items()})
        fh.write(json.dumps(line) + "\n"); fh.flush(); os.fsync(fh.fileno())
        done.add(f["fid"]); n_new += 1
        if n_new % 10 == 0:
            el = time.time() - t0
            log(f"frame {len(done)}/{len(frames)}  {el / n_new:.1f}s/frame  ETA {(len(frames) - len(done)) * el / n_new / 60:.0f} min")
        if len(done) % CFG["ckpt_every"] == 0:
            _snapshot_parquet(res_path)
    fh.close()
    _snapshot_parquet(res_path)
    log("main loop complete:", len(done), "frames")


def _snapshot_parquet(res_path):
    try:
        import pandas as pd
        rows = []
        for line in Path(res_path).read_text().splitlines():
            d = json.loads(line)
            for c, v in d["pred"].items():
                rows.append(dict(fid=d["fid"], ep=d["ep"], t=d["t"], cond=c, xyz=v["xyz"], grip=v["grip"], gripn=v["gripn"]))
        pd.DataFrame(rows).to_parquet(Path(res_path).with_suffix(".parquet"))
    except Exception as e:
        log("parquet snapshot skipped:", repr(e)[:80])


# ----------------------------------------------------------------------------- stage: video (GPU)
def stage_video(n_frames=48, sev=3):
    sel = jload(OUT / "selection.json")
    fpc = jload(OUT / "fp_check.json")
    pol = make_policy()
    pol.set_dtype(dtype_of(fpc["final_dtype"]))
    loader = make_loader(pol, sel)
    roles = Roles(pol, loader, sel)
    d = sel["episodes"][sel["hero_rank"]]
    ts = np.linspace(0, d["length"] - CFG["horizon"] - 1, n_frames).round().astype(int)
    W8 = CFG["win_short"]
    rec = dict(clean_img=[], smoke_img=[], p_clean=[], p_smoke=[], state=[])
    for i, t in enumerate(ts):
        obs = roles.obs(roles.traj(d, int(t)), int(t))
        seed = frame_seed(100000 + i)
        cx = infer(pol, roles, obs, seed)[0]
        cv, _ = corrupt_views(obs["video"], "SMOKE", sev, seed, which=[roles.endo])
        sx = infer(pol, roles, with_views(obs, cv), seed)[0]
        k = roles.endo
        sm = lambda a: cv2.resize(a[0, 0], (480, int(480 * a.shape[2] / a.shape[3])), interpolation=cv2.INTER_AREA)
        rec["clean_img"].append(sm(obs["video"][k])); rec["smoke_img"].append(sm(cv[k]))
        rec["p_clean"].append(cx[:W8]); rec["p_smoke"].append(sx[:W8])
        rec["state"].append(np.stack([obs["state"][kk][0, 0, :3] for kk in roles.pose]))
    scale = fpc["unit_scale"]
    np.savez_compressed(OUT / "video_data.npz", t=ts, clean_img=np.stack(rec["clean_img"]), smoke_img=np.stack(rec["smoke_img"]),
                        p_clean=np.stack(rec["p_clean"]) * scale, p_smoke=np.stack(rec["p_smoke"]) * scale,
                        state=np.stack(rec["state"]) * scale, sev=sev)
    log("video_data.npz saved")


# ----------------------------------------------------------------------------- analysis (CPU)
def load_results(path=None, meta=None):
    path = path or OUT / "results.jsonl"
    meta = meta or jload(OUT / "run_meta.json")
    conds = meta["conds"]
    ci = {c: i for i, c in enumerate(conds)}
    H, sc = meta["H"], meta["unit_scale"]
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    F, C = len(rows), len(conds)
    A, G = len(meta["pose_keys"]), len(meta["grip_keys"])
    XYZ = np.full((F, C, H, A, 3), np.nan, np.float32)
    GRN = np.full((F, C, H, G), np.nan, np.float32)
    GTX = np.zeros((F, H, A, 3), np.float32)
    PS = np.full((F, C), np.nan, np.float32)
    ep = np.zeros(F, int); fid = np.zeros(F, int); STA = np.zeros((F, A, 3), np.float32)
    for i, r in enumerate(rows):
        ep[i], fid[i] = r["ep"], r["fid"]
        GTX[i] = np.array(r["gt_xyz"], np.float32).reshape(H, A, 3) * sc
        STA[i] = np.array(r["state_xyz"], np.float32).reshape(A, 3) * sc
        for c, v in r["pred"].items():
            XYZ[i, ci[c]] = np.array(v["xyz"], np.float32).reshape(H, A, 3) * sc
            GRN[i, ci[c]] = np.array(v["gripn"], np.float32).reshape(H, G)
        for c, p in r["psnr"].items():
            PS[i, ci[c]] = p
    return dict(conds=conds, ci=ci, XYZ=XYZ, GRN=GRN, GTX=GTX, PS=PS, ep=ep, fid=fid, STA=STA, meta=meta, H=H)


WINDOWS = ("h8", "h50")


def _wlen(R, w):
    return min(CFG["win_short"], R["H"]) if w == "h8" else R["H"]


def per_frame_metrics(R):
    """NaN propagates (no nansum). Windows: mean over the first n steps of per-step distance over both arms."""
    X, ci = R["XYZ"], R["ci"]
    clean = X[:, ci["clean"]]

    def step_dist(a, b):
        return np.sqrt(np.sum((a - b) ** 2, axis=(-1, -2)))  # (...,H)

    dist = step_dist(X, clean[:, None])  # (F,C,H)
    gtd = step_dist(X, R["GTX"][:, None])
    M = {}
    for w in WINDOWS:
        n = _wlen(R, w)
        M["D_" + w] = dist[:, :, :n].mean(-1)
        M["G_" + w] = gtd[:, :, :n].mean(-1)
        M["E_" + w] = M["G_" + w][:, ci["clean"]]
        # mean GT displacement over the window (context for scale): per-step distance of GT from the first GT step
        M["GTdisp_" + w] = step_dist(R["GTX"][:, :n], R["GTX"][:, :1]).mean(-1)   # oracle: measured from GT step 0
        M["Hold_" + w] = step_dist(np.repeat(R["STA"][:, None], R["H"], axis=1)[:, :n], R["GTX"][:, :n]).mean(-1)   # hold the current state
    gn = R["GRN"]
    sgn = np.sign(gn)
    diff = (sgn != sgn[:, ci["clean"]][:, None]).astype(float)
    diff[np.isnan(gn) | np.isnan(gn[:, ci["clean"]][:, None])] = np.nan
    M["flip0"] = np.nanmean(diff[:, :, 0, :], axis=-1) if gn.shape[-1] else np.full(X.shape[:2], np.nan)
    n8 = _wlen(R, "h8")
    M["flip8"] = (np.nanmean(np.nanmax(diff[:, :, :n8, :], axis=2), axis=-1) if gn.shape[-1] else np.full(X.shape[:2], np.nan))
    return M


def _boot_weights(ep, B, rng):
    u, inv = np.unique(ep, return_inverse=True)
    n = len(u)
    idx = rng.integers(0, n, (B, n))
    W = np.stack([np.bincount(r, minlength=n) for r in idx]).astype(float)
    return W, inv, n


def _ep_sums(v, inv, n):
    v = np.asarray(v, float)
    ok = ~np.isnan(v)
    S = np.zeros((n,) + v.shape[1:]); N = np.zeros((n,) + v.shape[1:])
    np.add.at(S, inv, np.where(ok, v, 0)); np.add.at(N, inv, ok.astype(float))
    return S, N


def boot_mean(v, ep, B=None, seed=0):
    """Cluster bootstrap (episode) of the mean; NaN (frame, condition) pairs are excluded. v: (F,) or (F,C)."""
    B = B or CFG["bootstrap"]
    rng = np.random.default_rng(seed)
    W, inv, n = _boot_weights(ep, B, rng)
    S, N = _ep_sums(v, inv, n)
    m = S.sum(0) / np.maximum(N.sum(0), 1)
    bm = (W @ S) / np.maximum(W @ N, 1e-9)
    return m, np.percentile(bm, 2.5, axis=0), np.percentile(bm, 97.5, axis=0)


def boot_ratio(num, den, ep, B=None, seed=0):
    """Ratio of means D/E_clean over the same (frame, condition) pairs, episode-cluster bootstrap. num (F,C), den (F,)."""
    B = B or CFG["bootstrap"]
    rng = np.random.default_rng(seed)
    W, inv, n = _boot_weights(ep, B, rng)
    den2 = np.where(np.isnan(num) | np.isnan(den)[:, None], np.nan, den[:, None])
    num2 = np.where(np.isnan(den2), np.nan, num)
    Sn, _ = _ep_sums(num2, inv, n); Sd, _ = _ep_sums(den2, inv, n)
    r = Sn.sum(0) / np.maximum(Sd.sum(0), 1e-9)
    br = (W @ Sn) / np.maximum(W @ Sd, 1e-9)
    return r, np.percentile(br, 2.5, 0), np.percentile(br, 97.5, 0)


def exploratory(R, M):
    """EXPLORATORY metrics, not part of the locked verdict rules.

    (a) ensemble-mean drift: mean action over 4 seeds, corrupted vs clean, with a floor from splitting the clean seeds 2 vs 2;
    (b) vision-use fraction: artifact drift / blank-endoscope drift at the same seed (ratio of means);
    SANITY: 4-seed ensemble clean error vs hold-state error at the full horizon (NaN gives N/A).
    """
    ci, X, ep, sev = R["ci"], R["XYZ"], R["ep"], CFG["ensemble_sev"]
    cl = [X[:, ci[c]] for c in ("clean", "clean2", "clean3", "clean4")]
    ens_clean = np.mean(cl, axis=0)
    sd = lambda p, q: np.sqrt(((p - q) ** 2).sum(axis=(-1, -2)))
    ci_ = lambda v, seed: dict(zip(("mean", "lo", "hi"), (float(x) for x in boot_mean(v, ep, seed=seed))))
    out = dict(severity=sev, floor={}, ensemble_drift={}, vision_use={}, blank_endo_drift={})
    for w in WINDOWS:
        n = _wlen(R, w)
        out["floor"][w] = ci_(sd(np.mean(cl[:2], axis=0), np.mean(cl[2:], axis=0))[:, :n].mean(-1), 21)
        bd = M["D_" + w][:, ci["blank_endo"]]
        out["blank_endo_drift"][w] = ci_(bd, 24)
        for k in CFG["artifacts"]:
            ens_k = np.mean([X[:, ci[f"{k}|{sev}"]]] + [X[:, ci[f"{k}|{sev}|s{j}"]] for j in (1, 2, 3)], axis=0)
            out["ensemble_drift"][f"{w}|{k}"] = ci_(sd(ens_k, ens_clean)[:, :n].mean(-1), 22)
            r, lo, hi = boot_ratio(M["D_" + w][:, [ci[f"{k}|{sev}"]]], bd, ep, seed=23)
            out["vision_use"][f"{w}|{k}"] = dict(ratio=float(r[0]), lo=float(lo[0]), hi=float(hi[0]))
    ens_err, hold = float(np.nanmean(sd(ens_clean, R["GTX"]).mean(-1))), float(np.nanmean(M["Hold_h50"]))
    out["sanity"] = dict(ensemble_error=ens_err, hold_error=hold,
                         verdict="N/A" if np.isnan(ens_err) or np.isnan(hold) else ("PASS" if ens_err < hold else "FAIL"))
    return out


def analyze(R, verbose=True):
    M = per_frame_metrics(R)
    ci, ep = R["ci"], R["ep"]
    S, S2 = CFG["severities"], CFG["secondary_sevs"]
    out = dict(n_frames=int(len(ep)), n_episodes=int(len(np.unique(ep))), dtype=R["meta"]["dtype"], dataset=R["meta"]["base"],
               tag=R["meta"]["tag"], corrupted_view=R["meta"].get("corrupted_view"), H=R["H"])
    nan_pairs = {c: int(np.isnan(M["D_h8"][:, ci[c]]).sum() + np.isnan(M["D_h50"][:, ci[c]]).sum()) for c in R["conds"]}
    out["nan_pairs"] = {c: v for c, v in nan_pairs.items() if v}
    out["nan_pairs_total"] = int(sum(nan_pairs.values()))
    res, floors = {}, {}
    for w in WINDOWS:
        fm, flo, fhi = boot_mean(M["D_" + w][:, ci["clean2"]], ep, seed=1)
        floors[w] = dict(mean=float(fm), lo=float(flo), hi=float(fhi))
        em, _, _ = boot_mean(M["E_" + w], ep, seed=11)
        gd, _, _ = boot_mean(M["GTdisp_" + w], ep, seed=12)
        hs, _, _ = boot_mean(M["Hold_" + w], ep, seed=14)
        floors[w].update(clean_vs_gt=float(em), gt_displacement_oracle=float(gd), hold_state=float(hs))
        for k in CFG["kinds"]:
            cols = [ci[f"{k}|{s}"] for s in S]
            mean, lo, hi = boot_mean(M["D_" + w][:, cols], ep, seed=2)
            rr, rlo, rhi = boot_ratio(M["D_" + w][:, cols], M["E_" + w], ep, seed=3)
            gm, glo, ghi = boot_mean(M["G_" + w][:, cols], ep, seed=4)
            res[(w, k)] = dict(D=mean, Dlo=lo, Dhi=hi, R=rr, Rlo=rlo, Rhi=rhi, gt=gm, gtlo=glo, gthi=ghi)
            cols2 = [ci[f"{k}|{s}|all"] for s in S2]
            m2, l2, h2 = boot_mean(M["D_" + w][:, cols2], ep, seed=13)
            res[(w, k, "all")] = dict(D=m2, Dlo=l2, Dhi=h2)
    out["floor"] = floors
    flips = {}
    for key in ("flip0", "flip8"):
        for k in CFG["kinds"]:
            cols = [ci[f"{k}|{s}"] for s in S]
            flips[(key, k)] = boot_mean(M[key][:, cols], ep, seed=5)
        flips[(key, "floor")] = tuple(float(x) for x in boot_mean(M[key][:, ci["clean2"]], ep, seed=6))
    verdict, spec = {}, {}
    i3, i5 = S.index(3), S.index(5)
    for w in WINDOWS:
        fl = floors[w]["mean"]
        for k in CFG["kinds"]:
            r = res[(w, k)]
            fragile = bool(r["Dlo"][i3] > 2 * fl and r["R"][i3] > 0.5)
            robust = bool(r["D"][i5] < 2 * fl)
            verdict[(w, k)] = dict(fragile=fragile, robust=robust, label="FRAGILE" if fragile else ("ROBUST" if robust else "INTERMEDIATE"))
        per = []
        for s in S:
            ds = M["D_" + w][:, ci[f"SMOKE|{s}"]]
            mg, lg, _ = boot_mean(ds - M["D_" + w][:, ci[f"GAUSS|{s}"]], ep, seed=7)
            mb, lb, _ = boot_mean(ds - M["D_" + w][:, ci[f"BRIGHTNESS|{s}"]], ep, seed=8)
            per.append(dict(s=s, d_gauss=float(mg), lo_gauss=float(lg), d_bright=float(mb), lo_bright=float(lb), holds=bool(lg > 0 and lb > 0)))
        spec[w] = dict(per_severity=per, primary_s3=per[i3]["holds"], any=any(p["holds"] for p in per))
    out["psnr_by_sev"] = {k: np.nanmean(R["PS"][:, [ci[f"{k}|{s}"] for s in S]], axis=0).tolist() for k in CFG["kinds"]}
    out["verdict"] = {f"{w}|{k}": v for (w, k), v in verdict.items()}
    out["surgical_specific"] = spec
    out["res"] = {"|".join(key): {kk: np.asarray(vv).tolist() for kk, vv in v.items()} for key, v in res.items()}
    out["flips"] = {f"{a}|{b}": [np.asarray(x).tolist() for x in v] if isinstance(v, tuple) else v for (a, b), v in flips.items()}
    # hero frame: median over frames of the mean percentile rank of D8 at s=3 across the 5 artifacts
    pcts = np.zeros((len(ep), len(CFG["artifacts"])))
    for j, k in enumerate(CFG["artifacts"]):
        v = M["D_h8"][:, ci[f"{k}|3"]]
        ok = ~np.isnan(v)
        rk = np.full(len(v), np.nan)
        rk[ok] = (np.argsort(np.argsort(v[ok])) + 0.5) / ok.sum() * 100
        pcts[:, j] = rk
    mp = np.nanmean(pcts, axis=1)
    hi = int(np.nanargmin(np.abs(mp - 50)))
    out["hero"] = dict(index=hi, fid=int(R["fid"][hi]), mean_pct=float(mp[hi]), pct={k: float(pcts[hi, j]) for j, k in enumerate(CFG["artifacts"])})
    out["exploratory"] = exploratory(R, M)
    block = format_block(out, res, verdict, spec, flips)
    out["block"] = block
    if verbose:
        print(block)
    return out, M, res, flips


def format_block(out, res, verdict, spec, flips):
    S = CFG["severities"]
    i3, i5 = S.index(3), S.index(5)
    f8, f50 = out["floor"]["h8"], out["floor"]["h50"]
    L = ["", "=" * 78, "GR00T-H STRESS RESULT", "=" * 78,
         f"model {CFG['model_id']} | dtype {out['dtype']} | {out['tag']} | {out['dataset']}",
         f"{out['n_frames']} frames, {out['n_episodes']} episodes; corrupted view (primary) = {out['corrupted_view']}; CI = 95% bootstrap by episode ({CFG['bootstrap']})",
         "frames are GR00T-H TRAINING data; open-loop sensitivity only; corruptions are synthetic approximations",
         f"NaN (frame,condition) pairs excluded: {out['nan_pairs_total']}" + (f"  {out['nan_pairs']}" if out["nan_pairs_total"] else ""),
         "",
         f"{'':<22}{'8 steps':>12}{'50 steps':>12}",
         f"{'noise floor (mm)':<22}{f8['mean']:>12.3f}{f50['mean']:>12.3f}",
         f"{'clean vs GT (mm)':<22}{f8['clean_vs_gt']:>12.3f}{f50['clean_vs_gt']:>12.3f}",
         f"{'hold-state error (mm)':<22}{f8['hold_state']:>12.3f}{f50['hold_state']:>12.3f}",
         "", "D = translation drift vs clean, same seed (mm); R = D / E_clean (ratio of means); ROBUST = below sampling spread under shared noise",
         f"{'artifact':<12}{'D8 s3':>8}{'CI':>16}{'R8':>6}{'D8 s5':>8}{'D50 s3':>8}{'R50':>6}{'D50 s5':>8}{'flip8 s3':>9}  verdict(8 | 50)"]
    for k in CFG["kinds"]:
        a, b = res[("h8", k)], res[("h50", k)]
        va, vb = verdict[("h8", k)]["label"], verdict[("h50", k)]["label"]
        tag = f"{va} | {vb}" + ("" if k in CFG["artifacts"] else " (control)")
        L.append(f"{k:<12}{a['D'][i3]:>8.3f} [{a['Dlo'][i3]:>5.2f},{a['Dhi'][i3]:>5.2f}]{a['R'][i3]:>6.2f}{a['D'][i5]:>8.3f}{b['D'][i3]:>8.3f}{b['R'][i3]:>6.2f}{b['D'][i5]:>8.3f}"
                 f"{flips[('flip8', k)][0][i3]:>9.3f}  {tag}")
    L.append("")
    for k in CFG["artifacts"]:
        L.append(f"VERDICT {k:<10} 8-step: {verdict[('h8', k)]['label']:<13} (fragile@s3={verdict[('h8', k)]['fragile']}, robust@s5={verdict[('h8', k)]['robust']})"
                 f" | 50-step: {verdict[('h50', k)]['label']}")
    L.append("")
    for w, nm in (("h8", "8-step"), ("h50", "50-step")):
        sp = spec[w]
        sev_hold = [p["s"] for p in sp["per_severity"] if p["holds"]]
        L.append(f"VERDICT SMOKE [{nm}] {'SURGICAL-SPECIFIC' if sp['primary_s3'] else 'NOT SURGICAL-SPECIFIC'} at s=3: smoke exceeds Gaussian noise and brightness "
                 f"at matched MSE (model-input resolution), CI excl. 0; severities holding: {sev_hold or 'none'}")
    p3 = spec["h8"]["per_severity"][i3]
    L.append(f"  s3 8-step: SMOKE-GAUSS {p3['d_gauss']:+.3f} mm (CI lo {p3['lo_gauss']:+.3f}) | SMOKE-BRIGHT {p3['d_bright']:+.3f} mm (CI lo {p3['lo_bright']:+.3f})")
    L.append("  (BRIGHTNESS is a GR00T-H training augmentation, so low drift under it is expected)")
    L.append("  matched PSNR at model resolution (dB, s1..s5): " + ", ".join(f"{x:.1f}" for x in out["psnr_by_sev"]["SMOKE"]))
    L.append("  ALL-VIEWS secondary (D8 mm at s3 | s5): " + "; ".join(f"{k} {res[('h8', k, 'all')]['D'][0]:.2f}|{res[('h8', k, 'all')]['D'][1]:.2f}" for k in CFG["kinds"]))
    h = out["hero"]
    L.append(f"  hero frame fid {h['fid']}: mean percentile {h['mean_pct']:.0f}; " + ", ".join(f"{k} p{v:.0f}" for k, v in h["pct"].items()))
    e = out["exploratory"]
    sv = e["sanity"]
    L.append(f"SANITY {sv['verdict']}: 4-seed ensemble clean error {sv['ensemble_error']:.3f} mm vs hold-state error {sv['hold_error']:.3f} mm at {out['H']} steps (diagnostic only)")
    L += ["", f"EXPLORATORY (not part of the locked verdicts; primary condition, severity {e['severity']})",
          "  ensemble-mean drift (mm): mean over 4 seeds, corrupted vs clean; floor = clean seeds split 2 vs 2",
          f"  {'':<10}{'floor8':>8}{'floor50':>9}" + "".join(f"{k:>11}" for k in CFG["artifacts"])]
    for w, nm in (("h8", "8 steps"), ("h50", "50 steps")):
        L.append(f"  {nm:<10}{e['floor']['h8']['mean']:>8.3f}{e['floor']['h50']['mean']:>9.3f}" + "".join(f"{e['ensemble_drift'][f'{w}|{k}']['mean']:>11.3f}" for k in CFG["artifacts"]))
    L.append("  vision-use fraction = artifact drift / blank-endoscope drift, same seed (ratio of means, 95% CI)")
    for w, nm in (("h8", "8 steps"), ("h50", "50 steps")):
        L.append(f"  {nm:<10}" + "  ".join(f"{k} {e['vision_use'][f'{w}|{k}']['ratio']:.2f} [{e['vision_use'][f'{w}|{k}']['lo']:.2f},{e['vision_use'][f'{w}|{k}']['hi']:.2f}]" for k in CFG["artifacts"]))
    L.append("=" * 78)
    return "\n".join(L)


def stage_analysis():
    R = load_results()
    out, M, res, flips = analyze(R)
    jdump({k: v for k, v in out.items() if k != "block"}, OUT / "summary.json")
    (OUT / "result_block.txt").write_text(out["block"])
    import pandas as pd
    rows = []
    for key, r in res.items():
        w, k = key[0], key[1]
        sev = CFG["secondary_sevs"] if len(key) == 3 else CFG["severities"]
        for i, s in enumerate(sev):
            row = dict(window=w, kind=k, view="all" if len(key) == 3 else "endoscope", sev=s, D=r["D"][i], D_lo=r["Dlo"][i], D_hi=r["Dhi"][i])
            if len(key) == 2:
                row.update(R=r["R"][i], R_lo=r["Rlo"][i], R_hi=r["Rhi"][i], gt_err=r["gt"][i],
                           flip8=flips[("flip8", k)][0][i], flip0=flips[("flip0", k)][0][i],
                           psnr_model=out["psnr_by_sev"][k][i])
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "summary_table.csv", index=False)
    return out


# ----------------------------------------------------------------------------- figures (CPU)
STAMP = None   # text overlaid on every figure when set (selftest only)


def _stamp(fig):
    if STAMP:
        fig.text(0.5, 0.5, STAMP, rotation=25, ha="center", va="center", fontsize=30, color="#B00020", alpha=0.35, zorder=100)


def _style():
    import matplotlib as mpl
    from matplotlib import font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    fam = "Inter" if "Inter" in names else "DejaVu Sans"
    mpl.rcParams.update({
        "font.family": fam, "font.size": 10, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8, "grid.color": RULE, "grid.linewidth": 0.7,
        "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False})
    return fam


KIND_STYLE = {  # accent for SMOKE, inks for the rest, controls dashed
    "SMOKE": dict(color=ACCENT, lw=2.4, ls="-", marker="o"),
    "BLOOD": dict(color=INK, lw=1.4, ls="-", marker="o"),
    "FOG": dict(color="#4A5560", lw=1.4, ls="-", marker="s"),
    "GLARE": dict(color="#7D858F", lw=1.4, ls="-", marker="^"),
    "OCCLUSION": dict(color="#AAA79D", lw=1.4, ls="-", marker="D"),
    "GAUSS": dict(color="#4A5560", lw=1.2, ls=(0, (3, 2)), marker=None),
    "BRIGHTNESS": dict(color="#8A8F98", lw=1.2, ls=(0, (1, 2)), marker=None),
}


def _lab(k):
    return k.title() if k not in ("GAUSS", "BRIGHTNESS") else k.title() + " (control)"


def _line_fig(out, xs_by_kind, xlabel, path, window="h8", invert=False, title=None):
    import matplotlib.pyplot as plt
    _style()
    S = CFG["severities"]
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    fl = out["floor"][window]
    ax.axhspan(0, fl["mean"], color=RULE, alpha=0.9, lw=0)
    ax.axhline(2 * fl["mean"], color=GREY, lw=0.9, ls=(0, (4, 3)))
    ends = []
    allx = [np.min(v) for v in xs_by_kind.values()] + [np.max(v) for v in xs_by_kind.values()]
    x0, x1 = min(allx), max(allx)
    pad = 0.04 * (x1 - x0 + 1e-9)
    ax.text(x1 if invert else x0, fl["mean"] * 0.5, " noise floor (seed to seed)", va="center", ha="left", fontsize=8.5, color=GREY)
    ax.text(x1 if invert else x0, 2 * fl["mean"], " 2x floor", va="bottom", ha="left", fontsize=8.5, color=GREY)
    ctx = out["floor"][window]["clean_vs_gt"]
    for k in CFG["kinds"]:
        r = out["res"][f"{window}|{k}"]
        st = KIND_STYLE[k]
        x = np.array(xs_by_kind[k])
        D, lo, hi = np.array(r["D"]), np.array(r["Dlo"]), np.array(r["Dhi"])
        o = np.argsort(x)
        ax.fill_between(x[o], lo[o], hi[o], color=st["color"], alpha=0.12 if k != "SMOKE" else 0.18, lw=0)
        ax.plot(x[o], D[o], color=st["color"], lw=st["lw"], ls=st["ls"], marker=st["marker"], ms=4.5)
        ends.append((D[o][-1] if not invert else D[o][0], x[o][-1] if not invert else x[o][0], k))
    ends.sort()
    ymax = max(e[0] for e in ends) * 1.05
    ypos = []
    for d, xe, k in ends:
        y = d if not ypos else max(d, ypos[-1] + 0.045 * ymax)
        ypos.append(y)
        ax.text(xe + (-0.15 if invert else 0.15) * (x1 - x0) / 5, y, _lab(k), color=KIND_STYLE[k]["color"] if k != "OCCLUSION" else GREY,
                va="center", ha="left", fontsize=9, fontweight="bold" if k == "SMOKE" else "normal")
    ax.axhline(ctx, color=INK, lw=0.7, ls=(0, (1, 3)))
    ax.text(x0 if not invert else x1, ctx, " typical clean-vs-ground-truth error", fontsize=8, color=INK, va="bottom", ha="left")
    ax.set_ylim(0, max(ymax, ctx) * 1.12)
    ax.set_xlabel(xlabel); ax.set_ylabel(f"drift of predicted tool translation (mm, first {'8' if window == 'h8' else '50'} steps)")
    if invert:
        ax.set_xlim(x1 + pad, x0 - pad * 6)
    else:
        ax.set_xlim(x0 - pad, x1 + pad * 6)
    ax.grid(axis="x", visible=False)
    ax.set_title(title or "How far the predicted motion moves when the view is degraded", loc="left", fontsize=12, fontweight="bold", pad=24)
    ax.text(0, 1.03, f"GR00T-H-N1.7, {out['n_frames']} frames / {out['n_episodes']} episodes (training data), endoscope view corrupted. Mean and 95% CI over episodes.",
            transform=ax.transAxes, fontsize=8.3, color=GREY)
    fig.tight_layout(); _stamp(fig); fig.savefig(path, dpi=200); plt.close(fig)


def fig_drift(out, path, window="h8"):
    S = CFG["severities"]
    _line_fig(out, {k: S for k in CFG["kinds"]}, "severity", path, window)


def fig_drift_psnr(out, path, window="h8"):
    xs = {k: out["psnr_by_sev"][k] for k in CFG["kinds"]}
    _line_fig(out, xs, "PSNR of the corrupted view vs clean, at model input resolution (dB; milder to the left)", path, window, invert=True,
              title="Drift against how much the image changed")


def fig_flip(out, path):
    import matplotlib.pyplot as plt
    _style()
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    kinds = CFG["kinds"]
    x = np.arange(len(kinds))
    for off, si, alpha in ((-0.19, 2, 0.45), (0.19, 4, 1.0)):
        v = np.array([out["flips"][f"flip8|{k}"][0][si] for k in kinds]) * 100
        lo = np.array([out["flips"][f"flip8|{k}"][1][si] for k in kinds]) * 100
        hi = np.array([out["flips"][f"flip8|{k}"][2][si] for k in kinds]) * 100
        cols = [ACCENT if k == "SMOKE" else INK for k in kinds]
        ax.bar(x + off, v, 0.36, color=cols, alpha=alpha, lw=0)
        ax.errorbar(x + off, v, yerr=[np.maximum(v - lo, 0), np.maximum(hi - v, 0)], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2)
    fl = out["flips"]["flip8|floor"][0] * 100
    ax.axhline(fl, color=GREY, lw=0.9, ls=(0, (4, 3)))
    ax.text(-0.45, fl, "seed-to-seed floor", va="bottom", ha="left", fontsize=8.5, color=GREY)
    ax.set_xticks(x); ax.set_xticklabels([k.title() for k in kinds], fontsize=9)
    ax.set_ylabel("gripper sign flipped in any of first 8 steps (% frames)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Gripper open/close flips under corruption", loc="left", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.5, handles=[plt.Rectangle((0, 0), 1, 1, color=INK, alpha=0.45), plt.Rectangle((0, 0), 1, 1, color=INK)],
              labels=["severity 3", "severity 5"])
    fig.tight_layout(); _stamp(fig); fig.savefig(path, dpi=200); plt.close(fig)


def _rel_path(xyz_chunk, state_xyz, arm):
    """(H,A,3) absolute chunk -> (H+1,2) XY path relative to the current tool position (mm)."""
    p = xyz_chunk[:, arm, :2] - state_xyz[arm, :2]
    return np.vstack([[0, 0], p])


def _draw_path(ax, p, color, lw=2.0, ls="-", alpha=1.0):
    from matplotlib.patches import FancyArrowPatch
    ax.plot(p[:, 0], p[:, 1], color=color, lw=lw, ls=ls, alpha=alpha, solid_capstyle="round")
    if len(p) > 1 and np.linalg.norm(p[-1] - p[-2]) > 1e-9:
        ax.add_patch(FancyArrowPatch(p[-2], p[-1], arrowstyle="-|>", mutation_scale=9, color=color, lw=lw, alpha=alpha, shrinkA=0, shrinkB=0))


def fig_hero(panels, path, subtitle=""):
    """panels: list of (title, image, clean_chunk, corrupt_chunk or None, state_xyz). 2x3 grid of [image | path plot].
    The path plot sits beside the image (never over it): predicted XY path of both tools over the next 8 steps in the
    robot frame, mm. Clean white-on-dark ink, corrupted orange."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    _style()
    ext = 1e-6
    for _, _, cc, kc, st in panels:
        for ch in (cc, kc):
            if ch is None: continue
            for a in range(ch.shape[1]):
                ext = max(ext, np.abs(_rel_path(ch, st, a)).max())
    ext *= 1.15
    fig = plt.figure(figsize=(14.2, 6.6))
    gs = GridSpec(2, 6, figure=fig, width_ratios=[3.3, 1.25] * 3, left=0.01, right=0.995, bottom=0.02, top=0.83, wspace=0.05, hspace=0.20)
    for n, (title, im, cc, kc, st) in enumerate(panels):
        r, c = divmod(n, 3)
        ax = fig.add_subplot(gs[r, 2 * c]); ax.imshow(im); ax.set_axis_off(); ax.grid(False)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=ACCENT if kc is not None else INK, pad=5)
        ins = fig.add_subplot(gs[r, 2 * c + 1])
        ins.set_facecolor("#23262B"); ins.set_xlim(-ext, ext); ins.set_ylim(-ext, ext); ins.set_aspect("equal", anchor="C")
        ins.set_xticks([]); ins.set_yticks([]); ins.grid(False)
        for sp in ins.spines.values(): sp.set_visible(False)
        ins.axhline(0, color="white", lw=0.4, alpha=0.25); ins.axvline(0, color="white", lw=0.4, alpha=0.25)
        for a in range(cc.shape[1]):
            _draw_path(ins, _rel_path(cc, st, a), "white", lw=1.8, ls="-" if a == 0 else (0, (3, 2)))
            if kc is not None:
                _draw_path(ins, _rel_path(kc, st, a), ACCENT, lw=1.8, ls="-" if a == 0 else (0, (3, 2)))
        if n == 0:
            ins.text(0.5, -0.04, f"axes: +-{ext:.0f} mm", transform=ins.transAxes, ha="center", va="top", fontsize=7.5, color=GREY)
    fig.suptitle("One endoscope frame, five synthetic operating-room artifacts (severity 3)", x=0.01, y=1.0, ha="left", va="top", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.955, subtitle or "Dark panel: predicted tool path over the next 8 steps (robot-frame XY, mm). White = clean view, orange = corrupted view. Solid = PSM1, dashed = PSM2.",
             fontsize=9, color=GREY, ha="left", va="top")
    _stamp(fig); fig.savefig(path, dpi=170); plt.close(fig)


def fig_corruption_grid(img, path, seed=12345):
    """One frame x 7 corruptions x 5 severities (visual check). Labels show PSNR at model-input resolution."""
    import matplotlib.pyplot as plt
    _style()
    K, S = CFG["kinds"], CFG["severities"]
    fig, axes = plt.subplots(len(K), len(S) + 1, figsize=(2.55 * (len(S) + 1), 1.62 * len(K) + 0.5))
    for r, k in enumerate(K):
        axes[r, 0].imshow(img); axes[r, 0].set_ylabel(k.title(), rotation=0, ha="right", va="center", fontsize=10, fontweight="bold")
        for c, s in enumerate(S):
            out, _ = corrupt_view(img, k, s, seed, 0)
            axes[r, c + 1].imshow(out)
            axes[r, c + 1].set_xlabel(f"s{s}  {psnr_model(img, out):.1f} dB", fontsize=7.5, color=GREY, labelpad=1)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for sp in a.spines.values(): sp.set_visible(False)
    axes[0, 0].set_title("clean", fontsize=9, color=GREY)
    for c, s in enumerate(S): axes[0, c + 1].set_title(f"severity {s}", fontsize=9, color=GREY)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.96, bottom=0.01, wspace=0.03, hspace=0.12)
    _stamp(fig); fig.savefig(path, dpi=130); plt.close(fig)


def render_video(vd, path_mp4, path_gif=None, fps=6):
    """vd: dict from video_data.npz. Left clean view + white path, right SMOKE view + orange path, bottom trail."""
    import matplotlib.pyplot as plt
    import imageio.v2 as iio
    _style()
    n = len(vd["t"])
    pc, ps_, st = vd["p_clean"], vd["p_smoke"], vd["state"]
    ext = max(np.abs(_rel_path(pc[i], st[i], a)).max() for i in range(n) for a in range(pc.shape[2])) * 1.2 + 1e-6
    drift = np.array([np.linalg.norm((ps_[i] - pc[i]).reshape(pc.shape[1], -1), axis=1).mean() for i in range(n)])
    frames = []
    for i in range(n):
        fig = plt.figure(figsize=(9.6, 5.6), dpi=100)
        gs = fig.add_gridspec(2, 2, height_ratios=[3.1, 1.25], hspace=0.12, wspace=0.04, left=0.02, right=0.98, top=0.88, bottom=0.14)
        for j, (img, ch, col, ttl) in enumerate(((vd["clean_img"][i], pc[i], "white", "clean view"),
                                                  (vd["smoke_img"][i], ps_[i], ACCENT, f"smoke, severity {int(vd['sev'])} (endoscope)"))):
            ax = fig.add_subplot(gs[0, j]); ax.imshow(img); ax.set_axis_off(); ax.grid(False)
            ax.set_title(ttl, loc="left", fontsize=10.5, fontweight="bold", color=INK if j == 0 else ACCENT, pad=4)
            ins = ax.inset_axes([0.66, 0.04, 0.32, 0.32 * img.shape[1] / img.shape[0]])
            ins.set_facecolor((0.12, 0.13, 0.15, 0.62)); ins.set_xlim(-ext, ext); ins.set_ylim(-ext, ext); ins.set_aspect("equal")
            ins.set_xticks([]); ins.set_yticks([]); ins.grid(False)
            for sp in ins.spines.values(): sp.set_visible(False)
            for a in range(ch.shape[1]):
                _draw_path(ins, _rel_path(pc[i], st[i], a), "white", lw=1.5, alpha=0.9 if j == 0 else 0.45, ls="-" if a == 0 else (0, (3, 2)))
                if j == 1: _draw_path(ins, _rel_path(ch, st[i], a), ACCENT, lw=1.8, ls="-" if a == 0 else (0, (3, 2)))
        ax = fig.add_subplot(gs[1, :]); ax.plot(vd["t"], drift, color=RULE, lw=1.2)
        ax.plot(vd["t"][: i + 1], drift[: i + 1], color=ACCENT, lw=2); ax.scatter([vd["t"][i]], [drift[i]], color=ACCENT, s=22, zorder=3)
        ax.set_xlim(vd["t"][0], vd["t"][-1]); ax.set_ylim(0, max(drift.max() * 1.15, 1e-6)); ax.set_ylabel("drift (mm)", fontsize=8.5)
        ax.set_xlabel("frame in episode", fontsize=8.5); ax.tick_params(labelsize=8); ax.grid(axis="x", visible=False)
        fig.suptitle("Predicted tool path, clean vs smoke (same seed, first 8 steps)", x=0.02, y=0.975, ha="left", fontsize=12, fontweight="bold")
        _stamp(fig)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
    w = iio.get_writer(path_mp4, fps=fps, codec="libx264", quality=7, macro_block_size=1)
    for f in frames: w.append_data(f)
    w.close()
    if path_gif:
        small = [cv2.resize(f, (f.shape[1] * 2 // 3, f.shape[0] * 2 // 3), interpolation=cv2.INTER_AREA) for f in frames]
        iio.mimsave(path_gif, small, duration=1.0 / fps, loop=0)


def hero_panels(R, out, load_img):
    """Panels for the median-percentile frame; titles carry each artifact's D8 and that frame's percentile."""
    h = out["hero"]; i = h["index"]; ci = R["ci"]
    img = load_img(h["fid"])
    seed = frame_seed(h["fid"])
    W8 = min(CFG["win_short"], R["H"])
    cc, st = R["XYZ"][i, ci["clean"], :W8], R["STA"][i]
    panels = [("Clean", img, cc, None, st)]
    for k in CFG["artifacts"]:
        im, _ = corrupt_view(img, k, 3, seed, 0)
        kc = R["XYZ"][i, ci[f"{k}|3"], :W8]
        D = win_mean_dist(kc, cc, W8)
        panels.append((f"{k.title()}   drift {D:.1f} mm, p{h['pct'][k]:.0f} of 400 frames", im, cc, kc, st))
    sub = (f"This frame is the median across artifacts (mean percentile {h['mean_pct']:.0f}). Dark panel: predicted tool path, next 8 steps, robot-frame XY in mm.\n"
           "White = clean view, orange = corrupted view; solid = PSM1, dashed = PSM2.")
    return panels, sub


def stage_figures():
    R = load_results()
    out, M, res, flips = analyze(R, verbose=False)
    jdump({k: v for k, v in out.items() if k != "block"}, OUT / "summary.json")
    fig_drift(out, OUT / "fig_b_drift_vs_severity.png")
    fig_drift(out, OUT / "fig_b_drift_vs_severity_50step.png", window="h50")
    fig_drift_psnr(out, OUT / "fig_b2_drift_vs_psnr.png")
    fig_flip(out, OUT / "fig_c_gripper_flip.png")

    def load_img(fid):
        return cv2.imread(str(OUT / "frames" / f"{fid}.jpg"))[..., ::-1].copy()

    panels, sub = hero_panels(R, out, load_img)
    fig_hero(panels, OUT / "fig_a_hero.png", subtitle=sub)
    if (OUT / "video_data.npz").exists():
        vd = dict(np.load(OUT / "video_data.npz"))
        render_video(vd, str(OUT / "fig_d_episode.mp4"), str(OUT / "fig_d_episode.gif"))
    log("figures done")


def stage_zip():
    import shutil
    shutil.make_archive(str(WORK / "gr00th_stress_outputs"), "zip", OUT)
    log("zipped ->", WORK / "gr00th_stress_outputs.zip")


# ----------------------------------------------------------------------------- selftest (CPU, synthetic)
def synth_tissue(h=540, w=960, seed=3):
    """Pinkish tissue-like texture with vessels, instrument shafts and specular spots (fallback test frame)."""
    rng = np.random.default_rng(seed)
    f1, f2, f3 = (fractal_field(h, w, rng, octaves=o, base=b) for o, b in ((5, 3), (4, 6), (3, 10)))
    base = np.stack([0.80 - 0.10 * f1 - 0.05 * f3, 0.42 + 0.10 * f1 - 0.08 * f2, 0.40 + 0.08 * f2 - 0.06 * f1], -1)
    img = (base * 255).astype(np.uint8).copy()
    for _ in range(14):
        p = np.array([rng.uniform(0, w), rng.uniform(0, h)])
        d = rng.uniform(0, 2 * np.pi)
        pts = [p.copy()]
        for _ in range(40):
            d += rng.normal(0, 0.25); p = p + 14 * np.array([np.cos(d), np.sin(d)]); pts.append(p.copy())
        cv2.polylines(img, [np.array(pts, np.int32)], False, (150, 30, 40), int(rng.integers(2, 6)), cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    cv2.line(img, (int(w * 0.12), int(h * 0.95)), (int(w * 0.46), int(h * 0.42)), (150, 152, 158), 26, cv2.LINE_AA)
    cv2.line(img, (int(w * 0.88), int(h * 0.97)), (int(w * 0.62), int(h * 0.50)), (140, 144, 150), 22, cv2.LINE_AA)
    cv2.circle(img, (int(w * 0.46), int(h * 0.42)), 16, (190, 192, 198), -1, cv2.LINE_AA)
    return img


def test_images(outdir):
    """Synthetic tissue-like test frame (no network access, no third-party imagery)."""
    return [("synthetic", synth_tissue())]


def synth_results(path, meta_path, n_ep=10, per=10, seed=0):
    """Fake results.jsonl to exercise analysis + figures (NOT model output)."""
    rng = np.random.default_rng(seed)
    H, A = CFG["horizon"], 2
    conds = cond_names()
    gain = dict(SMOKE=1.0, BLOOD=0.6, FOG=0.5, GLARE=0.25, OCCLUSION=0.7, GAUSS=0.35, BRIGHTNESS=0.12)
    meta = dict(cfg_hash="synthetic", dtype="float16", unit_scale=1000.0, conds=conds, pose_keys=["psm1_pose", "psm2_pose"],
                grip_keys=["psm1_gripper", "psm2_gripper"], views=["endoscope_left"], corrupted_view="endoscope_left",
                tag="jhu_imerse_dvrk", base="SYNTHETIC", H=H, win_short=CFG["win_short"])
    jdump(meta, meta_path)
    fid = 0
    with open(path, "w") as fh:
        for e in range(n_ep):
            for j in range(per):
                base = np.cumsum(rng.normal(0, 0.0009, (H, A, 3)), 0) + rng.normal(0, 0.05, (A, 3))
                g0 = rng.choice([-0.8, 0.8], size=A)
                gt = base + rng.normal(0, 0.003, base.shape)
                pred = {}
                for c in conds:
                    parts = c.split("|")
                    if c.startswith("clean"): n_ = 0.0004
                    elif c == "blank_endo": n_ = 0.0035
                    else:
                        k, s = parts[0], int(parts[1])
                        n_ = 0.0004 + 0.0022 * gain[k] * (s / 3.0) ** 1.2 * (1.15 if parts[-1] == "all" else 1)
                    x = base + rng.normal(0, 1, base.shape) * n_ * np.linspace(0.6, 1.6, H)[:, None, None]
                    g = np.tile(g0, (H, 1)) + rng.normal(0, 0.1, (H, A))
                    if len(parts) >= 2 and parts[0] in gain and rng.random() < 0.25 * gain[parts[0]] * int(parts[1]) / 5: g = -g
                    pred[c] = dict(xyz=x.ravel().round(5).tolist(), grip=g.ravel().round(4).tolist(), gripn=g.ravel().round(4).tolist())
                ps = {f"{k}|{s}": round(30 - 3.5 * s, 2) for k in CFG["kinds"] for s in CFG["severities"]}
                fh.write(json.dumps(dict(fid=fid, ep=e, t=j * 20, gt_xyz=gt.ravel().round(5).tolist(), gt_grip=np.tile(g0, (H, 1)).ravel().tolist(),
                                         state_xyz=(base[0] - 0.001).ravel().round(5).tolist(), psnr=ps, pred=pred)) + "\n")
                fid += 1


def stage_selftest(outdir=None):
    """Exercise corruptions, analysis and figures on synthetic data. Everything is written to a temporary directory and stamped."""
    global OUT, STAMP
    STAMP = "SYNTHETIC SELFTEST"
    outdir = Path(outdir or tempfile.mkdtemp(prefix="gr00th_selftest_")); outdir.mkdir(parents=True, exist_ok=True)
    OUT = outdir / "_work"; OUT.mkdir(exist_ok=True)
    imgs = test_images(outdir)
    for name, im in imgs:
        cv2.imwrite(str(outdir / f"{name}.png"), im[..., ::-1])
    img = imgs[0][1]
    t0 = time.time()
    for name, im in imgs:
        fig_corruption_grid(im, outdir / ("corruption_grid.png" if name in ("tile_a", "synthetic") else f"corruption_grid_{name}.png"))
    log(f"corruption grids in {time.time() - t0:.1f}s")
    for k in CFG["kinds"]:
        t0 = time.time(); o, p = corrupt_view(img, k, 3, 7, 0)
        a, _ = corrupt_view(img, k, 3, 7, 0)
        log(f"{k:<10} s3 {time.time() - t0:5.2f}s  native PSNR {p:5.2f} dB  model-res PSNR {psnr_model(img, o):5.2f}  deterministic={np.array_equal(o, a)}")
    sm = [psnr_model(img, corrupt_view(img, "SMOKE", s, 7, 0)[0]) for s in CFG["severities"]]
    for k in ("GAUSS", "BRIGHTNESS"):
        d = [abs(psnr_model(img, corrupt_view(img, k, s, 7, 0)[0]) - sm[s - 1]) for s in CFG["severities"]]
        log(f"{k} model-res PSNR match error (dB) per severity:", np.round(d, 3))
    occ = [float(occlusion(img, s, _rng(7, 0, "OCCLUSION"), return_mask=True).mean()) for s in CFG["severities"]]
    log("occlusion shaft area fractions (target .05 .10 .20 .30 .40):", np.round(occ, 3))
    assert np.allclose(occ, [0.05, 0.10, 0.20, 0.30, 0.40], atol=0.01)
    # B1: the real observation layout (1,T=1,H,W,3), three views, endoscope-only vs all
    views = {"endoscope_left": img[None, None], "wrist_left": img[::-1, ::-1][None, None].copy(), "wrist_right": img[:, ::-1][None, None].copy()}
    for kind in CFG["kinds"]:
        o1, p1 = corrupt_views(views, kind, 3, 11, which=["endoscope_left"])
        o2, p2 = corrupt_views(views, kind, 3, 11, which=None)
        assert all(v.shape == (1, 1) + img.shape for v in o1.values()) and all(v.shape == (1, 1) + img.shape for v in o2.values())
        assert np.array_equal(o1["wrist_left"], views["wrist_left"]) and not np.array_equal(o2["wrist_left"], views["wrist_left"])
        assert np.isfinite(p1) and np.isfinite(p2)
    log("corrupt_views OK on (1,1,H,W,3): shapes preserved, wrists untouched in primary, all views changed in secondary")
    # analysis + figures on synthetic results
    synth_results(OUT / "results.jsonl", OUT / "run_meta.json")
    (OUT / "frames").mkdir(exist_ok=True)
    for f in range(100):
        cv2.imwrite(str(OUT / "frames" / f"{f}.jpg"), img[..., ::-1])
    R = load_results()
    # NaN handling check: poison one prediction, it must be excluded and counted, not scored as zero drift
    R["XYZ"][5, R["ci"]["SMOKE|3"]] = np.nan
    out, M, res, flips = analyze(R)
    assert out["nan_pairs_total"] == 2 and np.isnan(M["D_h8"][5, R["ci"]["SMOKE|3"]]), "NaN must propagate and be counted"
    fig_drift(out, outdir / "fig_b_drift_vs_severity.png")
    fig_drift_psnr(out, outdir / "fig_b2_drift_vs_psnr.png")
    fig_flip(out, outdir / "fig_c_gripper_flip.png")
    panels, sub = hero_panels(R, out, lambda fid: img)
    fig_hero(panels, outdir / "fig_a_hero.png", subtitle="synthetic predictions on a synthetic frame. " + sub)
    n = 24; rng = np.random.default_rng(1)
    base = np.cumsum(rng.normal(0, 1.2, (n, 8, 2, 3)), 1)
    vd = dict(t=np.arange(n) * 15, clean_img=np.stack([cv2.resize(img, (480, 270))] * n),
              smoke_img=np.stack([cv2.resize(corrupt_view(img, "SMOKE", 3, 7, 0)[0], (480, 270))] * n),
              p_clean=base, p_smoke=base + rng.normal(0, 1, base.shape) * np.linspace(0.3, 2.5, n)[:, None, None, None],
              state=np.zeros((n, 2, 3)), sev=3)
    render_video(vd, str(outdir / "fig_d_episode.mp4"), str(outdir / "fig_d_episode.gif"), fps=6)
    log("selftest outputs in", outdir)


# ----------------------------------------------------------------------------- CLI
STAGES = dict(data=stage_data, model_check=stage_model_check, diag=stage_diag, visual=stage_visual, main=stage_main, video=stage_video,
              analysis=stage_analysis, figures=stage_figures, zip=stage_zip)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--outdir", default=None, help="selftest output directory (default: a temporary directory)")
    a = ap.parse_args()
    log(f"stage {a.stage} | cfg hash {CFG_HASH} | work {WORK}")
    if a.stage == "selftest":
        stage_selftest(a.outdir)
    else:
        STAGES[a.stage]()

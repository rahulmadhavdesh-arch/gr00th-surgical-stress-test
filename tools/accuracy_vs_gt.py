"""Prediction error against recorded motion, paired by frame, with and without artifacts.

Error = mean over the first 8 predicted steps of the per-step Euclidean distance (mm) between the predicted and the recorded
end-effector translation, summed over both arms. Increase = error under a condition minus error on the clean frame, paired by frame.
95% CIs resample episodes (cluster bootstrap, 2000 draws).

Default mode reads results/per_frame_errors.csv and writes results/accuracy_vs_gt.json (run from anywhere):

    python tools/accuracy_vs_gt.py

The CSV is derived from the full per-frame run outputs, which are not in the repository (large). To rebuild it:

    python tools/accuracy_vs_gt.py --build-csv --results-jsonl PATH/results.jsonl --diag-jsonl PATH/diag.jsonl --meta PATH/run_meta.json

Columns: source (main = 400-frame sweep, diag = 60-frame subset), fid, ep, cond, err8 (error vs recorded motion, mm),
drift8 (distance to the clean prediction with the same seed, mm).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "per_frame_errors.csv"
JSON_PATH = ROOT / "results" / "accuracy_vs_gt.json"
WIN = 8
B = 2000
KINDS = ["SMOKE", "BLOOD", "FOG", "GLARE", "OCCLUSION", "GAUSS", "BRIGHTNESS"]
ARTIFACTS = KINDS[:5]
CLEAN_SEEDS = ["clean", "clean2", "clean3", "clean4"]


def needed_conds():
    c = list(CLEAN_SEEDS)
    for k in KINDS:
        c += [f"{k}|3", f"{k}|5"]
    for k in ARTIFACTS:
        c += [f"{k}|3|s{j}" for j in (1, 2, 3)]
    return c


# ------------------------------------------------------------------ build the CSV from the full run outputs
def _dist8(pred, ref):
    return np.sqrt(((pred[:WIN] - ref[:WIN]) ** 2).sum(axis=(-1, -2))).mean()


def build_csv(results_jsonl, diag_jsonl, meta_json, out=CSV_PATH):
    meta = json.load(open(meta_json))
    sc, H = meta["unit_scale"], meta["H"]
    A = len(meta["pose_keys"])
    want = set(needed_conds())
    rows = []
    for line in open(results_jsonl):
        if not line.strip():
            continue
        r = json.loads(line)
        gt = np.array(r["gt_xyz"], np.float32).reshape(H, A, 3) * sc
        clean = np.array(r["pred"]["clean"]["xyz"], np.float32).reshape(H, A, 3) * sc
        for c, v in r["pred"].items():
            if c not in want:
                continue
            p = np.array(v["xyz"], np.float32).reshape(H, A, 3) * sc
            rows.append(("main", r["fid"], r["ep"], c, _dist8(p, gt), _dist8(p, clean)))
    for line in open(diag_jsonl):
        if not line.strip():
            continue
        r = json.loads(line)
        gt = np.array(r["gt"], np.float32)
        clean = np.array(r["pred"]["clean"], np.float32)
        for c in ("clean", "blank_endo", "blank_all"):
            p = np.array(r["pred"][c], np.float32)
            rows.append(("diag", r["fid"], r["ep"], c, _dist8(p, gt), _dist8(p, clean)))
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "fid", "ep", "cond", "err8", "drift8"])
        for s, fid, ep, c, e, d in rows:
            w.writerow([s, fid, ep, c, "" if np.isnan(e) else f"{e:.5f}", "" if np.isnan(d) else f"{d:.5f}"])
    print(f"wrote {out} ({len(rows)} rows)")


# ------------------------------------------------------------------ analysis
def load_csv(path=CSV_PATH):
    """Returns {source: (ep array, fid array, {cond: {'err': arr, 'drift': arr}})}, rows aligned by fid within a source."""
    raw = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            raw.setdefault(r["source"], {}).setdefault(r["cond"], []).append(
                (int(r["fid"]), int(r["ep"]),
                 float(r["err8"]) if r["err8"] else np.nan, float(r["drift8"]) if r["drift8"] else np.nan))
    out = {}
    for s, conds in raw.items():
        ref = sorted(conds["clean"])
        fids = [x[0] for x in ref]
        ep = np.array([x[1] for x in ref])
        cd = {}
        for c, lst in conds.items():
            lst = sorted(lst)
            assert [x[0] for x in lst] == fids, f"frames differ for {c}"
            cd[c] = dict(err=np.array([x[2] for x in lst]), drift=np.array([x[3] for x in lst]))
        out[s] = (ep, np.array(fids), cd)
    return out


def boot_mean(v, ep, seed):
    """Mean with 95% episode-cluster bootstrap CI; NaN frames are excluded."""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(ep, return_inverse=True)
    n = len(u)
    W = np.stack([np.bincount(r, minlength=n) for r in rng.integers(0, n, (B, n))]).astype(float)
    ok = ~np.isnan(v)
    S = np.zeros(n); N = np.zeros(n)
    np.add.at(S, inv, np.where(ok, v, 0)); np.add.at(N, inv, ok.astype(float))
    bm = (W @ S) / np.maximum(W @ N, 1e-9)
    return dict(mean=float(S.sum() / max(N.sum(), 1)), lo=float(np.percentile(bm, 2.5)), hi=float(np.percentile(bm, 97.5)))


def analyze(data):
    ep, _, cd = data["main"]
    out = dict(
        n_frames=int(len(ep)), n_episodes=int(len(np.unique(ep))),
        definition=f"mean over the first {WIN} predicted steps of the per-step Euclidean distance (mm) to the recorded end-effector "
                   "translation, summed over both arms; increase = condition minus clean, paired by frame; "
                   f"95% CI from {B} episode-cluster bootstrap resamples",
        endoscope_view_only=True)
    out["clean_error"] = boot_mean(cd["clean"]["err"], ep, 0)
    out["single_seed"] = {}
    for k in KINDS:
        for s in (3, 5):
            c = f"{k}|{s}"
            r = boot_mean(cd[c]["err"] - cd["clean"]["err"], ep, 1)
            out["single_seed"][c] = dict(error=boot_mean(cd[c]["err"], ep, 1)["mean"], increase=r["mean"], lo=r["lo"], hi=r["hi"])
    cl4 = np.mean([cd[c]["err"] for c in CLEAN_SEEDS], axis=0)
    out["four_seed_avg_s3"] = {}
    for k in ARTIFACTS:
        a = np.mean([cd[f"{k}|3"]["err"]] + [cd[f"{k}|3|s{j}"]["err"] for j in (1, 2, 3)], axis=0)
        r = boot_mean(a - cl4, ep, 2)
        out["four_seed_avg_s3"][k] = dict(error=float(np.nanmean(a)), clean_error=float(np.nanmean(cl4)), increase=r["mean"], lo=r["lo"], hi=r["hi"])
    out["reseed_comparison_s3"] = {}
    rs = cd["clean2"]["drift"]
    for k in ARTIFACTS:
        a = cd[f"{k}|3"]["drift"]
        ok = ~(np.isnan(a) | np.isnan(rs))
        out["reseed_comparison_s3"][k] = dict(
            frac_drift_exceeds_own_reseed_drift=float((a[ok] > rs[ok]).mean()),
            mean_drift=float(a[ok].mean()), mean_reseed_drift=float(rs[ok].mean()))
    dep, _, dd = data["diag"]
    out["diag60"] = dict(n_frames=int(len(dep)), n_episodes=int(len(np.unique(dep))), clean_error=float(np.nanmean(dd["clean"]["err"])))
    for c in ("blank_endo", "blank_all"):
        r = boot_mean(dd[c]["err"] - dd["clean"]["err"], dep, 3)
        out["diag60"][c] = dict(error=float(np.nanmean(dd[c]["err"])), increase=r["mean"], lo=r["lo"], hi=r["hi"])
    return out


def print_table(o):
    print(f"Error vs recorded motion, first {WIN} steps, mm ({o['n_frames']} frames, {o['n_episodes']} episodes). Clean: {o['clean_error']['mean']:.2f}")
    print(f"{'condition':<16}{'s3 error':>9}{'s3 increase [95% CI]':>26}{'s5 error':>10}{'s5 increase [95% CI]':>26}")
    for k in KINDS:
        a, b = o["single_seed"][f"{k}|3"], o["single_seed"][f"{k}|5"]
        print(f"{k:<16}{a['error']:>9.2f}{a['increase']:>+10.2f} [{a['lo']:+.2f}, {a['hi']:+.2f}]{b['error']:>10.2f}{b['increase']:>+10.2f} [{b['lo']:+.2f}, {b['hi']:+.2f}]")
    print("4-seed average, s3:")
    for k, v in o["four_seed_avg_s3"].items():
        print(f"  {k:<14}{v['error']:>9.2f}{v['increase']:>+10.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}]")
    print("share of frames where drift exceeds that frame's clean-to-clean re-seed drift, s3:")
    print("  " + ", ".join(f"{k} {v['frac_drift_exceeds_own_reseed_drift']:.0%}" for k, v in o["reseed_comparison_s3"].items()))
    d = o["diag60"]
    print(f"60-frame subset: clean {d['clean_error']:.2f}; " + "; ".join(
        f"{c} {d[c]['error']:.2f} ({d[c]['increase']:+.2f} [{d[c]['lo']:+.2f}, {d[c]['hi']:+.2f}])" for c in ("blank_endo", "blank_all")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-csv", action="store_true")
    ap.add_argument("--results-jsonl"); ap.add_argument("--diag-jsonl"); ap.add_argument("--meta")
    a = ap.parse_args()
    if a.build_csv:
        build_csv(a.results_jsonl, a.diag_jsonl, a.meta)
    res = analyze(load_csv())
    JSON_PATH.write_text(json.dumps(res, indent=1) + "\n")
    print_table(res)

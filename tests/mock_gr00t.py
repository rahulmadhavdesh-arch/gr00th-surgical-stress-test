"""CPU mocks for the parts of the pipeline that need gr00t / HuggingFace / GPU. Used only when MOCK=1.

Provides: a fake HF dataset repo on disk (LeRobot v2.1 layout, tiny synthetic mp4s + parquet), a FakeApi that serves it,
a MockLoader (same call signature as LeRobotEpisodeLoader.get_sparse_episode), extract_step_data, and MockPolicy
(same call surface as the DTypePolicy built in make_policy: get_action, get_modality_config, set_dtype, last_norm, embodiment_tag).
"""
import json, os, shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

BASE = "Surgical/JHU/Imerse/NephFat_extracted/nephfat"
VIDEO_KEYS = ["observation.images.endoscope.left", "observation.images.wrist.left", "observation.images.wrist.right"]
STATE_LAYOUT = {"psm1_pose": (0, 7), "psm2_pose": (7, 14), "psm1_gripper": (14, 15), "psm2_gripper": (15, 16)}
ACTION_LAYOUT = {"psm1_pose": (0, 9), "psm2_pose": (9, 18), "psm1_gripper": (18, 19), "psm2_gripper": (19, 20)}
VIDEO_MOD = {"endoscope_left": VIDEO_KEYS[0], "wrist_left": VIDEO_KEYS[1], "wrist_right": VIDEO_KEYS[2]}
H, W = 270, 480


def repo_dir():
    return Path(os.environ["MOCK_REPO"])


def _tissue(rng):
    f = rng.random((H // 8, W // 8)).astype(np.float32)
    f = cv2.resize(f, (W, H), interpolation=cv2.INTER_CUBIC)
    return np.stack([0.75 - 0.2 * f, 0.4 + 0.2 * f, 0.4 + 0.1 * f], -1)


def build_fake_repo(root, n_eps=10):
    import pandas as pd
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    base = root / BASE
    (base / "meta").mkdir(parents=True)
    rng = np.random.default_rng(0)
    lengths = [140 + 7 * i for i in range(n_eps - 2)] + [60, 80]   # last two are too short (filter test)
    info = dict(codebase_version="v2.1", robot_type="dvrk", total_episodes=n_eps, total_frames=int(sum(lengths)), fps=30, chunks_size=1000,
                data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                features={k: {"dtype": "video", "shape": [H, W, 3]} for k in VIDEO_KEYS})
    (base / "meta" / "info.json").write_text(json.dumps(info))
    mod = dict(state={k: dict(start=a, end=b) for k, (a, b) in STATE_LAYOUT.items()},
               action={k: dict(start=a, end=b) for k, (a, b) in ACTION_LAYOUT.items()},
               video={k: dict(original_key=v) for k, v in VIDEO_MOD.items()}, annotation={"human.task_description": dict(original_key="task_index")})
    (base / "meta" / "modality.json").write_text(json.dumps(mod))
    (base / "meta" / "tasks.jsonl").write_text(json.dumps(dict(task_index=0, task="fake suturing task")) + "\n")
    # episode indices are not 0..n-1 so that pos != episode_index is exercised
    ep_ids = [3 + 2 * i for i in range(n_eps)]
    (base / "meta" / "episodes.jsonl").write_text("\n".join(json.dumps(dict(episode_index=e, tasks=["fake suturing task"], length=L)) for e, L in zip(ep_ids, lengths)) + "\n")
    for e, L in zip(ep_ids, lengths):
        ch = e // 1000
        tt = np.arange(L)[:, None]
        pos = np.stack([0.05 * np.sin(tt / 40 + a) + 0.01 * np.cumsum(rng.normal(0, .05, (L, 3)), 0) * 0 for a in (0, 1)], 0)  # (2,L,1)
        st, ac = np.zeros((L, 16), np.float32), np.zeros((L, 20), np.float32)
        for a, (s0, a0) in enumerate(((0, 0), (7, 9))):
            xyz = np.concatenate([0.05 * np.sin(tt / 40 + a + np.array([[0, 1, 2]])), ], 1)
            st[:, s0:s0 + 3] = xyz; st[:, s0 + 6] = 1.0
            ac[:, a0:a0 + 3] = xyz + 0.0015 * np.sin(tt / 7)
            ac[:, a0 + 3] = 1.0; ac[:, a0 + 7] = 1.0
        g = np.sign(np.sin(tt[:, 0] / 25.0)) * 0.8
        st[:, 14] = g; st[:, 15] = -g; ac[:, 18] = g; ac[:, 19] = -g
        df = pd.DataFrame({"observation.state": list(st), "action": list(ac), "timestamp": tt[:, 0] / 30.0, "frame_index": tt[:, 0],
                           "episode_index": e, "index": tt[:, 0], "task_index": 0})
        pth = base / info["data_path"].format(episode_chunk=ch, episode_index=e)
        pth.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pth)
        for vi, vk in enumerate(VIDEO_KEYS):
            vp = base / info["video_path"].format(episode_chunk=ch, episode_index=e, video_key=vk)
            vp.parent.mkdir(parents=True, exist_ok=True)
            wr = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
            tis = _tissue(rng)
            for t in range(L):
                img = np.roll(tis, t % 50, axis=1).copy()
                cv2.circle(img, (int(W * (0.3 + 0.2 * np.sin(t / 30 + vi))), int(H * 0.5)), 18, (0.8, 0.8, 0.85), -1)
                wr.write((img[..., ::-1] * 255).astype(np.uint8))
            wr.release()
    # decoy folders: a dataset without enough episodes (must be skipped) and a non-LeRobot dir
    d2 = root / "Surgical/JHU/suturebot/meta"; d2.mkdir(parents=True)
    (d2 / "info.json").write_text(json.dumps(dict(total_episodes=2, fps=30)))
    (root / "Surgical/JHU/notes").mkdir(parents=True); (root / "Surgical/JHU/notes/readme.txt").write_text("x")
    return root


# ------------------------------------------------------------------ fake HF api
class RepoFolder:
    def __init__(self, path): self.path = path


class RepoFile:
    def __init__(self, path, size): self.path, self.size = path, size


class FakeApi:
    def list_repo_tree(self, repo, path_in_repo=None, repo_type=None, recursive=False):
        d = repo_dir() / (path_in_repo or "")
        if not d.exists():
            raise FileNotFoundError(path_in_repo)
        out = []
        for p in sorted(d.iterdir()):
            rel = str(p.relative_to(repo_dir()))
            out.append(RepoFolder(rel) if p.is_dir() else RepoFile(rel, p.stat().st_size))
        return out


def fake_dl(repo, filename, repo_type=None, local_dir=None, **kw):
    src = repo_dir() / filename
    if not src.exists():
        raise FileNotFoundError(f"EntryNotFound {filename}")
    dst = Path(local_dir) / filename
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return str(dst)


# ------------------------------------------------------------------ loader (stand-in for LeRobotEpisodeLoader)
def _mc():
    mk = lambda keys: SimpleNamespace(modality_keys=list(keys), delta_indices=[0])
    return dict(video=mk(VIDEO_MOD), state=mk(STATE_LAYOUT), action=mk(ACTION_LAYOUT), language=mk(["annotation.human.task_description"]))


class MockLoader:
    def __init__(self, dataset_path, modality_configs, video_backend, video_backend_kwargs=None, require_stats=True, **kw):
        if video_backend != "opencv":
            raise ImportError(f"mock backend {video_backend} unavailable")   # exercises the backend fallback loop
        self.root = Path(dataset_path)
        self.modality_configs = deepcopy(modality_configs)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.mod = json.loads((self.root / "meta/modality.json").read_text())
        self.eps = [json.loads(l) for l in (self.root / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]

    def get_sparse_episode(self, pos, rows, video_indices=None, allow_padding=False):
        import pandas as pd
        e = int(self.eps[pos]["episode_index"]); ch = e // self.info["chunks_size"]
        df = pd.read_parquet(self.root / self.info["data_path"].format(episode_chunk=ch, episode_index=e))
        rows = np.asarray(rows)
        sub = df.iloc[rows].copy(); sub.index = rows
        S = np.stack(sub["observation.state"].values); A = np.stack(sub["action"].values)
        out = pd.DataFrame(index=rows)
        for k, v in self.mod["state"].items(): out[f"state.{k}"] = list(S[:, v["start"]:v["end"]])
        for k, v in self.mod["action"].items(): out[f"action.{k}"] = list(A[:, v["start"]:v["end"]])
        out["annotation.human.task_description"] = ["fake suturing task"] * len(rows)
        frames = {}
        for k, v in self.mod["video"].items():
            cap = cv2.VideoCapture(str(self.root / self.info["video_path"].format(episode_chunk=ch, episode_index=e, video_key=v["original_key"])))
            frames[k] = {}
            for t in (video_indices or []):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(t)); ok, im = cap.read()
                if not ok: raise RuntimeError(f"decode failed ep {e} t {t}")
                frames[k][int(t)] = im   # BGR, like the real opencv backend
            cap.release()
        out.attrs["frames"] = frames
        return out


def extract_step_data(traj, t, mc_obs, tag):
    fr = traj.attrs["frames"]
    images = {k: np.asarray(fr[k][int(t)])[None] for k in mc_obs["video"].modality_keys}
    states = {k: np.asarray(traj.loc[t, f"state.{k}"], np.float32)[None] for k in mc_obs["state"].modality_keys}
    return SimpleNamespace(images=images, states=states, text="fake suturing task")


# ------------------------------------------------------------------ policy
class MockPolicy:
    """Same surface as DTypePolicy. Output = last state xyz + seeded random walk + image-sensitive term (so corruption moves it)."""
    def __init__(self, tag):
        import torch
        self.embodiment_tag = tag; self.dtype = None; self.last_norm = None; self.mc = _mc(); self.model = torch.nn.Linear(2, 2)

    def get_modality_config(self): return self.mc

    def set_dtype(self, dt):
        self.dtype = dt

    def get_action(self, obs, options=None):
        import torch
        Hh = 50
        keys = self.mc["action"].modality_keys
        img = obs["video"][next(k for k in obs["video"] if "endo" in k)][0, 0].astype(np.float32) / 255.0
        sens = float(img.mean() - 0.5) + 0.3 * float(img.std())
        act, norm = {}, []
        for k in keys:
            d = 9 if "pose" in k else 1
            walk = torch.randn(Hh, d).cumsum(0) * 0.0004 + sens * 0.002 * torch.arange(Hh)[:, None] / Hh
            st = obs["state"][k.replace("action", "state")][0, 0] if k in obs["state"] else np.zeros(d)
            base = torch.zeros(Hh, d); base[:, : min(3, d)] = torch.tensor(np.asarray(st, np.float32)[: min(3, d)])
            a = (base + walk).to(self.dtype or torch.float32)
            act[k] = a.float().numpy()[None].astype(np.float32); norm.append(a.float().numpy())
        self.last_norm = np.concatenate(norm, -1)[None].astype(np.float32)
        return act, {}


def make_mock_policy(tag):
    return MockPolicy(tag)

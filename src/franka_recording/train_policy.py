"""Train and serve an ACT-style visuomotor policy on LLM-driven Franka stacking episodes.

    train_policy.py train --data <dataset_dir> --out <run_dir>
    train_policy.py serve --checkpoint <run_dir>/policy.pt [--address 127.0.0.1:47710]

Inputs: scene and wrist RGB (size and scene crop come from the dataset's ``image_spec``, default 128x128), the 9
measured joint positions, the task (which block goes on which), and with ``--state-obs`` the ground-truth block
positions.
Output: a chunk of the next ``--chunk`` actions at 15 Hz. With ``--action goal`` (default) an action is the
grasp-point goal the LLM's current move_to primitive was driving to plus a gripper open command, so RMPflow
reproduces the demonstrated motion; ``--action ee`` predicts the next hand position instead, and
``--action joint`` the 9 absolute joint position targets.
Each camera is encoded by a ResNet-18 into a grid of tokens (stride 32, e.g. 4x4 at 128 px, 20x15 at 640x480); the image, joint and task tokens go through a
transformer encoder, and a transformer decoder turns ``chunk`` learned queries into actions (L1 loss on
normalized targets). Only successful episodes are used for training.

``serve`` answers requests from ``franka_llm_episodes.py --mode policy`` over multiprocessing.connection.
"""

import argparse
import json
import math
import os
import time
from multiprocessing.connection import Listener

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image as PILImage

NUM_TASKS = 6
ACTION_DIM = 9
STATE_DIM = 9  # red, green, blue block positions


def episode_actions(ep: dict, action: str) -> np.ndarray:
    """Per-frame action targets: joint position targets, or the next hand position + gripper open command."""
    if action == "joint":
        return ep["actions"]
    if action == "goal":
        return np.concatenate([ep["goal_position"], ep["gripper_command"][:, None]], axis=1).astype(np.float32)
    hand = np.concatenate([ep["ee_pose"][1:, :3], ep["ee_pose"][-1:, :3]])
    gripper = np.concatenate([ep["gripper_command"][1:], ep["gripper_command"][-1:]])
    return np.concatenate([hand, gripper[:, None]], axis=1).astype(np.float32)


def episode_state(ep: dict) -> np.ndarray:
    return ep["block_poses"][:, :, :3].reshape(len(ep["block_poses"]), -1)


def feature_grid(height: int, width: int) -> int:
    """Number of ResNet-18 feature tokens (stride 32) for an image of this size."""
    for _ in range(5):
        height, width = (height + 1) // 2, (width + 1) // 2
    return height * width


class Policy(nn.Module):
    def __init__(self, chunk: int, action_dim: int = ACTION_DIM, state_obs: bool = False, dim: int = 256,
                 pretrained: bool = True, image_shapes=((128, 128), (128, 128))):
        super().__init__()
        self.chunk = chunk
        self.state_obs = state_obs
        self.encoders = nn.ModuleList()
        for _ in range(2):
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = torchvision.models.resnet18(weights=weights)
            self.encoders.append(nn.Sequential(*list(resnet.children())[:-2]))
        self.image_proj = nn.Conv2d(512, dim, 1)
        self.joint_proj = nn.Linear(ACTION_DIM, dim)
        self.task_embed = nn.Embedding(NUM_TASKS, dim)
        self.state_proj = nn.Linear(STATE_DIM, dim) if state_obs else None
        self.camera_embed = nn.Parameter(torch.zeros(2, 1, dim))
        num_tokens = sum(feature_grid(h, w) for h, w in image_shapes) + 2 + int(state_obs)
        self.position_embed = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, chunk, dim) * 0.02)
        self.transformer = nn.Transformer(d_model=dim, nhead=8, num_encoder_layers=4, num_decoder_layers=4,
                                          dim_feedforward=1024, dropout=0.1, batch_first=True, norm_first=True)
        self.head = nn.Linear(dim, action_dim)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, scene: torch.Tensor, wrist: torch.Tensor, joints: torch.Tensor, task: torch.Tensor,
                state: torch.Tensor | None = None):
        tokens = []
        for i, image in enumerate((scene, wrist)):
            x = (image.float() / 255.0 - self.image_mean) / self.image_std
            features = self.image_proj(self.encoders[i](x))  # (B, dim, H/32, W/32)
            tokens.append(features.flatten(2).transpose(1, 2) + self.camera_embed[i])
        tokens.append(self.joint_proj(joints).unsqueeze(1))
        tokens.append(self.task_embed(task).unsqueeze(1))
        if self.state_obs:
            tokens.append(self.state_proj(state).unsqueeze(1))
        memory = torch.cat(tokens, dim=1) + self.position_embed
        queries = self.queries.expand(scene.shape[0], -1, -1)
        return self.head(self.transformer(memory, queries))


def load_dataset(data_dir: str, action: str):
    episodes = []
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".npz"):
            continue
        with np.load(os.path.join(data_dir, name)) as d:
            if not bool(d["success"]) or len(d["timestamps"]) < 10:
                continue
            ep = {k: d[k] for k in d.files if k not in ("success", "seed", "task_index", "image_spec")}
            size = d["scene_rgb"].shape[1]
            ep["image_spec"] = json.loads(str(d["image_spec"])) if "image_spec" in d.files else \
                {"scene": {"crop": None, "size": [size, size]}, "wrist": {"crop": None, "size": [size, size]}}
            ep |= {"task_index": int(d["task_index"]), "name": name}
        ep["targets"] = episode_actions(ep, action)
        ep["state"] = episode_state(ep).astype(np.float32)
        episodes.append(ep)
    return episodes


class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, episodes, chunk: int, stats: dict):
        self.episodes = episodes
        self.chunk = chunk
        self.stats = stats
        self.index = [(e, t) for e, ep in enumerate(episodes) for t in range(len(ep["targets"]))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e, t = self.index[i]
        ep = self.episodes[e]
        steps = np.minimum(np.arange(t, t + self.chunk), len(ep["targets"]) - 1)
        actions = (ep["targets"][steps] - self.stats["action_mean"]) / self.stats["action_std"]
        joints = (ep["joint_positions"][t] - self.stats["joint_mean"]) / self.stats["joint_std"]
        state = (ep["state"][t] - self.stats["state_mean"]) / self.stats["state_std"]
        return (torch.from_numpy(ep["scene_rgb"][t]).permute(2, 0, 1), torch.from_numpy(ep["wrist_rgb"][t]).permute(2, 0, 1),
                torch.from_numpy(joints.astype(np.float32)), torch.tensor(ep["task_index"]),
                torch.from_numpy(state.astype(np.float32)), torch.from_numpy(actions.astype(np.float32)))


def augment(images: torch.Tensor) -> torch.Tensor:
    """Random shift (pad + crop, 1/16 of the image) and brightness/contrast jitter, batched on the GPU."""
    b, c, h, w = images.shape
    pad = max(4, round(min(h, w) / 16))
    padded = F.pad(images.float(), (pad, pad, pad, pad), mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (b,), device=images.device)
    oy = torch.randint(0, 2 * pad + 1, (b,), device=images.device)
    rows = (oy.view(b, 1) + torch.arange(h, device=images.device).view(1, h))
    cols = (ox.view(b, 1) + torch.arange(w, device=images.device).view(1, w))
    shifted = padded[torch.arange(b, device=images.device).view(b, 1, 1, 1), torch.arange(c, device=images.device).view(1, c, 1, 1),
                     rows.view(b, 1, h, 1), cols.view(b, 1, 1, w)]
    brightness = torch.empty(b, 1, 1, 1, device=images.device).uniform_(-20, 20)
    contrast = torch.empty(b, 1, 1, 1, device=images.device).uniform_(0.85, 1.15)
    mean = shifted.mean(dim=(1, 2, 3), keepdim=True)
    return ((shifted - mean) * contrast + mean + brightness).clamp(0, 255)


def train(args) -> None:
    torch.manual_seed(0)
    episodes = load_dataset(args.data, args.action)
    if len(episodes) < 2:
        raise SystemExit(f"need at least 2 successful episodes in {args.data}, found {len(episodes)}")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(episodes))
    num_val = max(1, int(round(args.val_fraction * len(episodes))))
    val_eps = [episodes[i] for i in order[:num_val]]
    train_eps = [episodes[i] for i in order[num_val:]]
    all_actions = np.concatenate([e["targets"] for e in train_eps])
    all_joints = np.concatenate([e["joint_positions"] for e in train_eps])
    all_states = np.concatenate([e["state"] for e in train_eps])
    stats = {"action_mean": all_actions.mean(0), "action_std": all_actions.std(0) + 1e-3,
             "joint_mean": all_joints.mean(0), "joint_std": all_joints.std(0) + 1e-3,
             "state_mean": all_states.mean(0), "state_std": all_states.std(0) + 1e-3}
    train_set, val_set = ChunkDataset(train_eps, args.chunk, stats), ChunkDataset(val_eps, args.chunk, stats)
    print(f"{len(train_eps)} train episodes ({len(train_set)} frames), {len(val_eps)} val episodes "
          f"({len(val_set)} frames)", flush=True)
    loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                         drop_last=True, persistent_workers=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, num_workers=4)

    device = torch.device("cuda")
    image_spec = train_eps[0]["image_spec"]
    image_shapes = [train_eps[0][f"{camera}_rgb"].shape[1:3] for camera in ("scene", "wrist")]
    print(f"images: scene {image_shapes[0]}, wrist {image_shapes[1]}, spec {image_spec}", flush=True)
    model = Policy(args.chunk, action_dim=all_actions.shape[1], state_obs=args.state_obs,
                   image_shapes=image_shapes).to(device)
    backbone = [p for n, p in model.named_parameters() if n.startswith("encoders")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoders")]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": args.lr * 0.1}, {"params": rest, "lr": args.lr}],
                                  weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train_log.jsonl"), "w")

    def evaluate() -> float:
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for batch in val_loader:
                for scene, wrist, joints, task, state, actions in zip(*(t.split(micro_batch) for t in batch)):
                    pred = model(scene.to(device), wrist.to(device), joints.to(device), task.to(device),
                                 state.to(device))
                    total += F.l1_loss(pred.float(), actions.to(device), reduction="sum").item()
                    count += actions.numel()
        model.train()
        return total / count

    def accumulate(batch) -> float:
        """Forward/backward the batch in micro-batches; gradients equal those of the full batch."""
        total = 0.0
        for scene, wrist, joints, task, state, actions in zip(*(t.split(micro_batch) for t in batch)):
            scene, wrist = augment(scene.to(device, non_blocking=True)), augment(wrist.to(device, non_blocking=True))
            joints = joints.to(device) + 0.01 * torch.randn_like(joints.to(device))
            state = state.to(device) + 0.01 * torch.randn_like(state.to(device))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(scene, wrist, joints, task.to(device), state)
            loss = F.l1_loss(pred.float(), actions.to(device)) * len(actions) / args.batch_size
            loss.backward()
            total += loss.item()
        return total

    # largest micro-batch that fits next to whatever else is on the GPU
    micro_batch = args.micro_batch or args.batch_size
    probe = next(iter(loader))
    while True:
        try:
            optimizer.zero_grad(set_to_none=True)
            accumulate(probe)
            break
        except torch.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if micro_batch == 1:
                raise
            micro_batch //= 2
    print(f"micro-batch {micro_batch} x {args.batch_size // micro_batch} accumulation steps", flush=True)

    step, start, best = 0, time.time(), float("inf")
    while step < args.steps:
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss_value = accumulate(batch)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            step += 1
            if step % 100 == 0 or step == args.steps:
                record = {"step": step, "loss": round(loss_value, 5), "elapsed_s": round(time.time() - start)}
                if step % 1000 == 0 or step == args.steps:
                    record["val_l1"] = round(evaluate(), 5)
                    checkpoint = {"model": model.state_dict(), "chunk": args.chunk, "action": args.action,
                                  "state_obs": args.state_obs, "action_dim": int(all_actions.shape[1]),
                                  "stats": {k: v.tolist() for k, v in stats.items()}, "step": step,
                                  "image_size": int(train_eps[0]["scene_rgb"].shape[1]), "image_spec": image_spec,
                                  "train_episodes": [e["name"] for e in train_eps],
                                  "val_episodes": [e["name"] for e in val_eps]}
                    torch.save(checkpoint, os.path.join(args.out, "policy_last.pt"))
                    if record["val_l1"] < best:
                        best = record["val_l1"]
                        torch.save(checkpoint, os.path.join(args.out, "policy.pt"))
                print(json.dumps(record), flush=True)
                log.write(json.dumps(record) + "\n")
                log.flush()
            if step >= args.steps:
                break


def serve(args) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    action = checkpoint.get("action", "joint")
    state_obs = checkpoint.get("state_obs", False)
    size = checkpoint["image_size"]
    image_spec = checkpoint.get("image_spec") or {camera: {"crop": None, "size": [size, size]}
                                                   for camera in ("scene", "wrist")}
    image_shapes = [(image_spec[c]["size"][1], image_spec[c]["size"][0]) for c in ("scene", "wrist")]
    model = Policy(checkpoint["chunk"], action_dim=checkpoint.get("action_dim", ACTION_DIM), state_obs=state_obs,
                   pretrained=False, image_shapes=image_shapes).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    stats = {k: np.asarray(v, dtype=np.float32) for k, v in checkpoint["stats"].items()}
    host, port = args.address.split(":")
    listener = Listener((host, int(port)), authkey=b"franka-policy")
    print(f"serving {args.checkpoint} (step {checkpoint['step']}) on {args.address}", flush=True)

    def image(rgb, camera: str):
        spec = image_spec[camera]
        if spec.get("crop"):
            x0, y0, x1, y1 = spec["crop"]
            rgb = rgb[y0:y1, x0:x1]
        width, height = spec["size"]
        if rgb.shape[:2] != (height, width):
            rgb = np.asarray(PILImage.fromarray(rgb).resize((width, height), PILImage.BILINEAR))
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).to(device)

    while True:
        connection = listener.accept()
        print("client connected", flush=True)
        try:
            while True:
                obs = connection.recv()
                joints = (np.asarray(obs["joint_positions"], np.float32) - stats["joint_mean"]) / stats["joint_std"]
                state = None
                if state_obs:
                    state = (np.asarray(obs["block_positions"], np.float32).reshape(-1) - stats["state_mean"]) \
                        / stats["state_std"]
                    state = torch.from_numpy(state).unsqueeze(0).to(device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(image(obs["scene_rgb"], "scene"), image(obs["wrist_rgb"], "wrist"),
                                 torch.from_numpy(joints).unsqueeze(0).to(device),
                                 torch.tensor([int(obs["task_index"])], device=device), state)
                actions = pred.float().cpu().numpy()[0] * stats["action_std"] + stats["action_mean"]
                key = {"joint": "joint_targets", "ee": "ee_targets", "goal": "goal_targets"}[action]
                connection.send({key: actions[: args.execute]})
        except (EOFError, ConnectionResetError):
            print("client disconnected", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--micro-batch", type=int, default=0,
                   help="Split each batch into micro-batches of this size (0: largest that fits in GPU memory)")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--action", choices=("goal", "ee", "joint"), default="goal",
                   help="goal: the LLM's current move_to grasp-point goal + gripper open command (RMPflow moves to "
                        "it exactly as during data collection); ee: next hand positions + gripper command; "
                        "joint: absolute joint position targets")
    p.add_argument("--state-obs", action="store_true",
                   help="Also observe the ground-truth block positions (the LLM sees these too)")
    p = sub.add_parser("serve")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--address", default="127.0.0.1:47710")
    p.add_argument("--execute", type=int, default=8, help="Actions of each predicted chunk to execute")
    args = parser.parse_args()
    train(args) if args.command == "train" else serve(args)


if __name__ == "__main__":
    main()

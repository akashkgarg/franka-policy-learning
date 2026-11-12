#!/usr/bin/env python3
"""Franka LLM Lab: browse LLM-driven and policy-driven Franka episodes, and launch new runs.

Run through run.sh (sources ROS 2 Jazzy for rosbag2_py). Serves:
    /                       the web app (web/)
    /api/episodes           episode summaries from every source directory
    /api/episode/<src>/<n>  episode.json + a timeline extracted from the bag (hand, fingers, blocks, goals)
    /api/video/<src>/<n>    scene | wrist camera video (H.264, generated from the bag on first request)
    /api/stats              dataset-level statistics
    /api/lab/...            run controller: start runs on a persistent Isaac Sim worker, poll live state
"""

from __future__ import annotations

import argparse
import base64
import glob
import http.server
import json
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
import urllib.parse
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RECORDING_DIR = HERE.parent / "franka_recording"
ISAAC_SIM_DIR = Path(os.environ.get("ISAAC_SIM_DIR", HERE.parent.parent))
WEB_DIR = HERE / "web"
GRASP_OFFSET_Z = 0.1034
COLORS = ("red", "green", "blue")
TIMELINE_HZ = 30.0
VIDEO_VERSION = 1
TIMELINE_VERSION = 2

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--port", type=int, default=8240)
parser.add_argument("--data", default=os.path.expanduser("~/franka_llm_data"))
parser.add_argument("--gpu", type=int, default=1, help="GPU for the lab sim worker and policy servers")
parser.add_argument("--policy-python", default=os.path.expanduser("~/.venvs/franka-policy/bin/python"))
parser.add_argument("--token-file", default=os.path.expanduser("~/.config/llm-gateway/lca_token"))
args = parser.parse_args()

DATA = Path(args.data)
CACHE = DATA / ".lab_cache"
LAB = DATA / "lab"
for d in (CACHE, LAB / "queue", LAB / "runs", LAB / "status", LAB / "live"):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------------------------------------------
# Episode index


def sources() -> dict[str, Path]:
    found = {"llm": DATA / "episodes", "vision_wood": DATA / "vision" / "wood_episodes",
             "vision_grid": DATA / "vision" / "grid_episodes", "vision_pilot": DATA / "vision" / "pilot",
             "vision_wood_pilot": DATA / "vision" / "wood_pilot", "lab": LAB / "runs"}
    for path in sorted(DATA.glob("eval_*")):
        if path.is_dir():
            found[path.name] = path
    if (DATA / "smoke").is_dir():
        found["smoke"] = DATA / "smoke"
    return {k: v for k, v in found.items() if v.is_dir()}


_index_lock = threading.Lock()
_index: dict[str, dict] = {}
_index_mtimes: dict[str, float] = {}
_index_time = 0.0


def has_bag(directory: Path) -> bool:
    return any(directory.glob("*.mcap"))


def summarize(source: str, directory: Path, episode: dict) -> dict:
    steps = episode.get("steps") or []
    latencies = [s["llm"]["latency_s"] for s in steps if isinstance(s.get("llm"), dict)]
    return {
        "id": f"{source}/{directory.name}",
        "source": source,
        "name": directory.name,
        "mode": episode.get("mode"),
        "observation": episode.get("observation") or ("state" if episode.get("mode") == "llm" else None),
        "num_locates": sum(1 for s in steps if (s.get("decision") or {}).get("action") == "locate"),
        "model": episode.get("model"),
        "instruction": episode.get("instruction") or f"Stack the {episode['task']['top']} block on top of the "
                                                         f"{episode['task']['bottom']} block.",
        "top": episode["task"]["top"],
        "bottom": episode["task"]["bottom"],
        "seed": episode.get("seed"),
        "success": episode.get("success"),
        "xy_error_m": episode.get("xy_error_m"),
        "num_steps": episode.get("num_steps"),
        "sim_seconds": episode.get("sim_seconds"),
        "wall_seconds": episode.get("wall_seconds"),
        "error": episode.get("error"),
        "llm_latency_s": round(sum(latencies), 1) if latencies else None,
        "has_bag": has_bag(directory),
        "has_llm_images": (directory / "llm_images").is_dir(),
        "table": episode.get("table") or "grid",
        "mtime": directory.stat().st_mtime,
    }


def refresh_index(force: bool = False) -> dict[str, dict]:
    global _index_time
    with _index_lock:
        if not force and time.time() - _index_time < 10:
            return _index
        seen = set()
        for source, root in sources().items():
            for path in root.glob("*/episode.json"):
                key = str(path)
                seen.add(key)
                mtime = path.stat().st_mtime
                if _index_mtimes.get(key) == mtime:
                    continue
                try:
                    with open(path) as f:
                        episode = json.load(f)
                    _index[key] = summarize(source, path.parent, episode)
                    _index_mtimes[key] = mtime
                except Exception:
                    traceback.print_exc()
        for key in list(_index):
            if key not in seen:
                del _index[key]
                _index_mtimes.pop(key, None)
        _index_time = time.time()
        return _index


def episode_dir(source: str, name: str) -> Path:
    root = sources().get(source)
    if root is None or "/" in name or name.startswith("."):
        raise FileNotFoundError(f"{source}/{name}")
    directory = root / name
    if not (directory / "episode.json").is_file():
        raise FileNotFoundError(f"{source}/{name}")
    return directory


# ---------------------------------------------------------------------------------------------------------------
# Bag extraction (timeline + video), cached under <data>/.lab_cache


_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)


def open_reader(directory: Path, topics: list[str]):
    import rosbag2_py

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(directory), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    return reader


def stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def build_timeline(directory: Path, episode: dict) -> dict:
    from geometry_msgs.msg import PoseStamped
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image, JointState

    topics = {"/ee_pose": PoseStamped, "/joint_states": JointState, "/joint_commands": JointState,
              "/scene_camera/camera_info": None}
    topics |= {f"/blocks/{c}/pose": PoseStamped for c in COLORS}
    reader = open_reader(directory, [t for t in topics if topics[t] is not None] + ["/scene_camera/rgb"])
    series = defaultdict(list)
    frame_times = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == "/scene_camera/rgb":
            # only the header is needed; deserializing the full image is the slow part but unavoidable here
            frame_times.append(stamp(deserialize_message(data, Image)))
            continue
        msg = deserialize_message(data, topics[topic])
        t = stamp(msg)
        if topic == "/ee_pose" or topic.startswith("/blocks/"):
            p = msg.pose.position
            series[topic].append((t, (p.x, p.y, p.z)))
        else:
            names = list(msg.name)
            fingers = [msg.position[names.index(n)] for n in ("panda_finger_joint1", "panda_finger_joint2")
                       if n in names]
            arm = [msg.position[names.index(f"panda_joint{i}")] for i in range(1, 8) if f"panda_joint{i}" in names]
            series[topic].append((t, (*arm, *fingers)))
    if not series["/ee_pose"]:
        raise RuntimeError("bag has no /ee_pose")
    t0 = series["/ee_pose"][0][0]
    t1 = series["/ee_pose"][-1][0]
    times = np.arange(t0, t1 + 1e-6, 1.0 / TIMELINE_HZ)

    def sample(topic: str) -> np.ndarray:
        data = series.get(topic) or []
        if not data:
            return np.full((len(times), 3), np.nan)
        keys = np.array([d[0] for d in data])
        values = np.array([d[1] for d in data], dtype=np.float64)
        idx = np.clip(np.searchsorted(keys, times + 1e-6) - 1, 0, len(keys) - 1)
        return values[idx]

    hand = sample("/ee_pose")
    joints = sample("/joint_states")
    commands = sample("/joint_commands")
    timeline = {
        "t": np.round(times - t0, 4).tolist(),
        "t0": t0,
        "grasp": np.round(hand - [0.0, 0.0, GRASP_OFFSET_Z], 4).tolist(),
        "fingers": np.round(joints[:, 7:9], 4).tolist() if joints.shape[1] >= 9 else [],
        "arm_joints": np.round(joints[:, :7], 4).tolist() if joints.shape[1] >= 7 else [],
        "gripper_open_cmd": (commands[:, 7] > 0.02).astype(int).tolist() if commands.shape[1] >= 9 else [],
        "blocks": {c: np.round(sample(f"/blocks/{c}/pose"), 4).tolist() for c in COLORS},
        "video_times": np.round(np.array(frame_times) - t0, 4).tolist(),
    }
    # LLM steps: each primitive runs from (sim_time - duration) to sim_time; thinking takes no sim time
    segments = []
    for step in episode.get("steps") or []:
        decision = step.get("decision")
        if not decision:
            continue
        result = step.get("result") or {}
        end = step["sim_time"] - t0
        start = end - float(result.get("duration_s") or 0.0)
        target = None
        if decision.get("action") == "move_to":
            target = result.get("clipped_target") or [decision["x"], decision["y"], decision["z"]]
        segments.append({"index": step["index"], "action": decision["action"], "start": round(start, 4),
                         "end": round(end, 4), "target": target})
    timeline["segments"] = segments
    policy_targets = [(s["sim_time"] - t0, s["first_target"]) for s in episode.get("steps") or [] if "first_target" in s]
    timeline["policy_targets"] = [[round(t, 3), *v] for t, v in policy_targets]
    return timeline


def cached_timeline(directory: Path, episode: dict) -> dict | None:
    if not has_bag(directory):
        return None
    key = f"{directory.parent.name}__{directory.name}"
    path = CACHE / f"{key}.timeline.v{TIMELINE_VERSION}.json"
    with _locks[key + "t"]:
        if path.exists() and path.stat().st_mtime >= (directory / "episode.json").stat().st_mtime:
            return json.loads(path.read_text())
        timeline = build_timeline(directory, episode)
        path.write_text(json.dumps(timeline))
        return timeline


def build_video(directory: Path, out: Path) -> None:
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image

    reader = open_reader(directory, ["/scene_camera/rgb", "/wrist_camera/rgb"])
    frames: dict[float, dict[str, np.ndarray]] = defaultdict(dict)
    process = None
    pending: list[float] = []
    width = height = None
    tmp = out.with_suffix(".tmp.mp4")

    def emit(t: float) -> None:
        nonlocal process, width, height
        pair = frames.pop(t)
        image = np.concatenate([pair["/scene_camera/rgb"], pair["/wrist_camera/rgb"]], axis=1)
        if process is None:
            height, width = image.shape[:2]
            process = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                 "-r", "15", "-i", "-", "-vf", "scale=1280:-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)], stdin=subprocess.PIPE)
        process.stdin.write(np.ascontiguousarray(image).tobytes())

    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, Image)
        rgb = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, -1)[..., :3]
        t = round(stamp(msg), 4)
        frames[t][topic] = rgb
        if len(frames[t]) == 2:
            pending.append(t)
        pending.sort()
        while pending and len(pending) > 2:
            emit(pending.pop(0))
    for t in pending:
        emit(t)
    if process is None:
        raise RuntimeError("bag has no camera frames")
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    os.replace(tmp, out)


def cached_video(directory: Path) -> Path:
    key = f"{directory.parent.name}__{directory.name}"
    path = CACHE / f"{key}.v{VIDEO_VERSION}.mp4"
    with _locks[key + "v"]:
        if not path.exists() or path.stat().st_mtime < (directory / "episode.json").stat().st_mtime:
            build_video(directory, path)
    return path


# ---------------------------------------------------------------------------------------------------------------
# Dataset statistics


_stats_cache: dict = {"time": 0.0, "data": None}
VISION_SOURCES = ("vision_wood", "vision_grid", "vision_pilot", "vision_wood_pilot")


def compute_stats() -> dict:
    if _stats_cache["data"] is not None and time.time() - _stats_cache["time"] < 60:
        return _stats_cache["data"]
    index = refresh_index()
    by_source = defaultdict(lambda: {"episodes": 0, "successes": 0})
    for e in index.values():
        by_source[e["source"]]["episodes"] += 1
        by_source[e["source"]]["successes"] += int(bool(e["success"]))
    llm_root = sources().get("llm")
    latencies, steps_hist, xy_errors, picks, places, actions = [], Counter(), [], [], [], Counter()
    per_task = Counter()
    tokens = {"prompt": 0, "completion": 0}
    approach_heights, place_heights = [], []
    reasoning_samples = []
    if llm_root:
        for path in llm_root.glob("*/episode.json"):
            with open(path) as f:
                episode = json.load(f)
            steps = episode.get("steps") or []
            steps_hist[len(steps)] += 1
            xy_errors.append(episode.get("xy_error_m"))
            per_task[f"{episode['task']['top']} on {episode['task']['bottom']}"] += 1
            last_move = None
            for s in steps:
                decision = s.get("decision") or {}
                action = decision.get("action")
                actions[action] += 1
                if isinstance(s.get("llm"), dict):
                    latencies.append(s["llm"]["latency_s"])
                    tokens["prompt"] += s["llm"].get("prompt_tokens") or 0
                    tokens["completion"] += s["llm"].get("completion_tokens") or 0
                if action == "move_to":
                    last_move = [decision["x"], decision["y"], decision["z"]]
                    if s["index"] == 0:
                        approach_heights.append(decision["z"])
                elif action == "close_gripper" and last_move:
                    picks.append([*last_move[:2], episode["task"]["top"]])
                elif action == "open_gripper" and last_move:
                    places.append([*last_move[:2], episode["task"]["bottom"]])
                    place_heights.append(last_move[2])
            if len(reasoning_samples) < 40 and len(steps) > 9:
                reasoning_samples.append({"episode": path.parent.name, "steps": [
                    {"action": (s.get("decision") or {}).get("action"), "reasoning": (s.get("decision") or {}).get("reasoning")}
                    for s in steps]})
    vision = {"steps_hist": Counter(), "locates_hist": Counter(), "locate_errors": [], "locate_cameras": Counter(),
              "xy_errors": [], "latencies": []}
    for source in VISION_SOURCES:
        root = sources().get(source)
        if not root:
            continue
        for path in root.glob("*/episode.json"):
            with open(path) as f:
                episode = json.load(f)
            steps = episode.get("steps") or []
            vision["steps_hist"][len(steps)] += 1
            vision["xy_errors"].append(episode.get("xy_error_m"))
            locates = [s for s in steps if (s.get("decision") or {}).get("action") == "locate"]
            vision["locates_hist"][len(locates)] += 1
            for s in locates:
                vision["locate_cameras"][s["decision"].get("camera")] += 1
                if s.get("debug"):
                    vision["locate_errors"].append([s["decision"].get("camera"), s["debug"]["xy_error_to_top_center_m"]])
            vision["latencies"] += [s["llm"]["latency_s"] for s in steps if isinstance(s.get("llm"), dict)]
    vision["steps_hist"] = dict(sorted(vision["steps_hist"].items()))
    vision["locates_hist"] = dict(sorted(vision["locates_hist"].items()))
    data = {
        "vision": vision,
        "by_source": by_source,
        "llm": {"latencies": latencies, "steps_hist": dict(sorted(steps_hist.items())), "xy_errors": xy_errors,
                "picks": picks, "places": places, "actions": actions, "per_task": per_task, "tokens": tokens,
                "approach_heights": approach_heights, "place_heights": place_heights,
                "long_episodes": reasoning_samples},
    }
    _stats_cache.update(time=time.time(), data=data)
    return data


# ---------------------------------------------------------------------------------------------------------------
# Run controller: one persistent Isaac Sim worker (franka_llm_episodes.py --jobs) + policy servers


class Lab:
    def __init__(self):
        self.lock = threading.Lock()
        self.worker: subprocess.Popen | None = None
        self.policy_servers: dict[str, tuple[subprocess.Popen, str]] = {}
        self.started_at = None

    def checkpoints(self) -> dict[str, dict]:
        found = {}
        for path in sorted(DATA.glob("runs/*/policy.pt")):
            log = path.parent / "train_log.jsonl"
            info = {"path": str(path), "name": path.parent.name}
            if log.exists():
                lines = [json.loads(l) for l in log.read_text().splitlines() if "val_l1" in l]
                if lines:
                    info["val_l1"] = min(l["val_l1"] for l in lines)
            summary = DATA / f"eval_{path.parent.name}" / "summary.json"
            if summary.exists():
                s = json.loads(summary.read_text())
                info["eval"] = f"{s['successes']}/{s['episodes']}"
            meta_vision_only = "vision" in path.parent.name
            info["inputs"] = "images + joints + task" + ("" if meta_vision_only else " + block positions")
            found[path.parent.name] = info
        return found

    def worker_alive(self) -> bool:
        return self.worker is not None and self.worker.poll() is None

    def worker_ready(self) -> bool:
        info = LAB / "worker.json"
        if not self.worker_alive() or not info.exists():
            return False
        try:
            return json.loads(info.read_text()).get("pid") is not None and info.stat().st_mtime >= (self.started_at or 0)
        except Exception:
            return False

    def ensure_worker(self) -> None:
        with self.lock:
            if self.worker_alive():
                return
            (LAB / "worker.json").unlink(missing_ok=True)
            env = dict(os.environ)
            env.update({"ROS_DOMAIN_ID": "90", "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
                        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
                        "VK_DRIVER_FILES": env.get("VK_DRIVER_FILES", "/etc/vulkan/icd.d/nvidia_icd.json")})
            for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "AMENT_PREFIX_PATH", "PYTHONHOME", "ROS_DISTRO",
                        "ROS_VERSION", "ROS_PYTHON_VERSION", "COLCON_PREFIX_PATH", "CMAKE_PREFIX_PATH"):
                env.pop(key, None)
            log = open(LAB / "worker.log", "a")
            self.started_at = time.time()
            self.worker = subprocess.Popen(
                [str(ISAAC_SIM_DIR / "python.sh"), str(RECORDING_DIR / "franka_llm_episodes.py"), "--jobs", str(LAB),
                 "--gpu", str(args.gpu)], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def ensure_policy(self, name: str) -> str:
        checkpoint = self.checkpoints().get(name)
        if checkpoint is None:
            raise ValueError(f"unknown policy {name}")
        with self.lock:
            server = self.policy_servers.get(name)
            if server and server[0].poll() is None:
                return server[1]
            port = 47750 + len(self.policy_servers)
            address = f"127.0.0.1:{port}"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
            for key in ("PYTHONPATH", "LD_LIBRARY_PATH"):
                env.pop(key, None)
            log = open(LAB / f"policy_{name}.log", "a")
            process = subprocess.Popen([args.policy_python, str(RECORDING_DIR / "train_policy.py"), "serve",
                                        "--checkpoint", checkpoint["path"], "--address", address],
                                       env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.policy_servers[name] = (process, address)
        for _ in range(120):
            if "serving" in (LAB / f"policy_{name}.log").read_text():
                return address
            if process.poll() is not None:
                raise RuntimeError(f"policy server {name} exited; see {LAB}/policy_{name}.log")
            time.sleep(0.5)
        raise RuntimeError(f"policy server {name} did not start")

    def submit(self, request: dict) -> dict:
        controller = request.get("controller", "llm")
        top, bottom = request.get("top"), request.get("bottom")
        if top not in COLORS or bottom not in COLORS or top == bottom:
            raise ValueError("pick two different block colors")
        seed = request.get("seed")
        seed = int(seed) if seed not in (None, "") else int(np.random.default_rng().integers(1_000_000, 9_999_999))
        job_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
        table = request.get("table") or "grid"
        if table not in ("grid", "wood", "plain"):
            raise ValueError(f"unknown table {table}")
        job = {"id": job_id, "top": top, "bottom": bottom, "seed": seed, "table": table, "submitted": time.time()}
        if controller in ("llm", "llm-vision"):
            observation = "vision" if controller == "llm-vision" else "state"
            job.update(mode="llm", observation=observation, model=request.get("model") or "gpt-6-astra",
                       instruction=(request.get("instruction") or "").strip() or None,
                       max_steps=int(request.get("max_steps") or (40 if observation == "vision" else 24)))
        elif controller.startswith("policy:"):
            name = controller.split(":", 1)[1]
            self.ensure_worker()
            job.update(mode="policy", policy=name, policy_address=self.ensure_policy(name),
                       policy_seconds=float(request.get("policy_seconds") or 30))
        else:
            raise ValueError(f"unknown controller {controller}")
        self.ensure_worker()
        job_file = LAB / "queue" / f"{job_id}.json"
        job_file.with_suffix(".tmp").write_text(json.dumps(job))
        os.replace(job_file.with_suffix(".tmp"), job_file)
        (LAB / "status" / f"{job_id}.json").write_text(json.dumps({"id": job_id, "state": "queued", "job": job,
                                                                   "events": []}))
        return job

    def status(self) -> dict:
        jobs = []
        for path in sorted((LAB / "status").glob("*.json"), reverse=True)[:30]:
            try:
                s = json.loads(path.read_text())
                jobs.append({"id": s["id"], "state": s["state"], "job": s["job"],
                             "success": (s.get("result") or {}).get("success")})
            except Exception:
                pass
        return {"worker": "ready" if self.worker_ready() else ("starting" if self.worker_alive() else "stopped"),
                "gpu": args.gpu, "policies": self.checkpoints(), "jobs": jobs, "token": token_info()}

    def stop(self) -> None:
        for process in [self.worker, *(p for p, _ in self.policy_servers.values())]:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, 15)
                except ProcessLookupError:
                    pass


lab = Lab()


def token_info() -> dict:
    try:
        token = open(args.token_file).read().strip()
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return {"present": True, "expires": claims.get("exp"), "expired": claims.get("exp", 0) < time.time(),
                "subject": claims.get("sub")}
    except Exception:
        return {"present": False}


# ---------------------------------------------------------------------------------------------------------------
# HTTP


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, content_type: str, cache: bool = False):
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        if range_header:
            m = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
        self.send_response(206 if range_header else 200)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self):
        try:
            self.route_get(urllib.parse.urlparse(self.path))
        except FileNotFoundError as exc:
            self.send_json({"error": f"not found: {exc}"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            traceback.print_exc()
            self.send_json({"error": repr(exc)}, 500)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            path = urllib.parse.urlparse(self.path).path
            if path == "/api/lab/run":
                self.send_json(lab.submit(body))
            elif path == "/api/lab/token":
                token = (body.get("token") or "").strip()
                if token.count(".") != 2:
                    raise ValueError("that does not look like a JWT")
                os.makedirs(os.path.dirname(args.token_file), mode=0o700, exist_ok=True)
                with open(args.token_file, "w") as f:
                    f.write(token)
                os.chmod(args.token_file, 0o600)
                self.send_json(token_info())
            elif path == "/api/lab/start":
                lab.ensure_worker()
                self.send_json({"ok": True})
            else:
                self.send_json({"error": "not found"}, 404)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, 400)
        except Exception as exc:
            traceback.print_exc()
            self.send_json({"error": repr(exc)}, 500)

    def route_get(self, url):
        path = url.path
        if path == "/api/episodes":
            self.send_json(sorted(refresh_index().values(), key=lambda e: (e["source"] != "lab", -e["mtime"])))
            return
        if path == "/api/stats":
            self.send_json(compute_stats())
            return
        if path == "/api/lab/status":
            self.send_json(lab.status())
            return
        m = re.match(r"^/api/episode/([^/]+)/([^/]+)$", path)
        if m:
            directory = episode_dir(m.group(1), m.group(2))
            episode = json.loads((directory / "episode.json").read_text())
            timeline = None
            timeline_error = None
            try:
                timeline = cached_timeline(directory, episode)
            except Exception as exc:
                traceback.print_exc()
                timeline_error = repr(exc)
            self.send_json({"episode": episode, "timeline": timeline, "timeline_error": timeline_error,
                            "summary": summarize(m.group(1), directory, episode)})
            return
        m = re.match(r"^/api/video/([^/]+)/([^/]+)\.mp4$", path)
        if m:
            directory = episode_dir(m.group(1), m.group(2))
            if not has_bag(directory):
                raise FileNotFoundError("no bag recorded for this episode")
            self.send_file(cached_video(directory), "video/mp4", cache=True)
            return
        m = re.match(r"^/api/llm_image/([^/]+)/([^/]+)/(step\d+_(?:scene|wrist)\.jpg)$", path)
        if m:
            image = episode_dir(m.group(1), m.group(2)) / "llm_images" / m.group(3)
            if not image.is_file():
                raise FileNotFoundError(m.group(3))
            self.send_file(image, "image/jpeg", cache=True)
            return
        m = re.match(r"^/api/lab/job/([\w-]+)$", path)
        if m:
            status_path = LAB / "status" / f"{m.group(1)}.json"
            status = json.loads(status_path.read_text())
            live = LAB / "live" / m.group(1) / "state.json"
            if live.exists():
                try:
                    status["live"] = json.loads(live.read_text())
                except json.JSONDecodeError:
                    pass
            self.send_json(status)
            return
        m = re.match(r"^/api/lab/live/([\w-]+)/(scene|wrist)\.jpg$", path)
        if m:
            image = LAB / "live" / m.group(1) / f"{m.group(2)}.jpg"
            if not image.exists():
                raise FileNotFoundError("no frame yet")
            self.send_file(image, "image/jpeg")
            return
        # static files
        relative = path.lstrip("/") or "index.html"
        file = (WEB_DIR / relative).resolve()
        if not str(file).startswith(str(WEB_DIR)) or not file.is_file():
            file = WEB_DIR / "index.html"
        types = {".html": "text/html", ".js": "application/javascript", ".css": "text/css", ".svg": "image/svg+xml"}
        self.send_file(file, types.get(file.suffix, "application/octet-stream"), cache=file.name == "plotly.min.js")


def main() -> None:
    refresh_index(force=True)
    print(f"{len(_index)} episodes from {', '.join(sources())}", flush=True)
    server = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    server.daemon_threads = True
    print(f"Franka LLM Lab on http://localhost:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        lab.stop()


if __name__ == "__main__":
    main()

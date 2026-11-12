"""Convert recorded Franka episodes (ROS 2 MCAP bags) into a compact imitation-learning dataset.

Run with the system ROS 2 Jazzy Python:
    bash -c 'source /opt/ros/jazzy/setup.bash; python3.12 bags_to_dataset.py <episodes_dir> <dataset_dir>'

One ``<episode>.npz`` is written per bag, sampled at the camera frame times (15 Hz):
    scene_rgb, wrist_rgb   uint8 (T, H, W, 3)   camera images, optionally cropped, then resized
    image_spec             JSON string          per camera: {"crop": [x0, y0, x1, y1] | null, "size": [W, H]}
    joint_positions        float32 (T, 9)       measured panda_joint1..7 + both finger joints
    joint_velocities       float32 (T, 9)
    actions                float32 (T, 9)       commanded joint position targets at the frame time
    ee_pose                float32 (T, 7)       tool (panda_hand) position + wxyz orientation
    block_poses            float32 (T, 3, 7)    red, green, blue block position + wxyz orientation
    gripper_command        float32 (T,)         1 = commanded open, 0 = commanded closed
    goal_position          float32 (T, 3)       grasp-point goal of the move_to primitive being executed
    timestamps             float64 (T,)         simulation time
    task_index, success, seed                   episode metadata (from episode.json)
An ``index.json`` summarizes every converted episode.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image as PILImage

JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
BLOCK_COLORS = ("red", "green", "blue")
GRASP_OFFSET_Z = 0.1034  # panda_hand origin to grasp point with the gripper pointing down


def stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def preprocess(rgb: np.ndarray, spec: dict) -> np.ndarray:
    """Crop (in source pixels) and resize one camera image; train_policy.py serve applies the same spec."""
    if spec.get("crop"):
        x0, y0, x1, y1 = spec["crop"]
        rgb = rgb[y0:y1, x0:x1]
    width, height = spec["size"]
    if rgb.shape[1] == width and rgb.shape[0] == height:
        return np.ascontiguousarray(rgb)
    return np.asarray(PILImage.fromarray(rgb).resize((width, height), PILImage.BILINEAR))


def convert(bag_dir: str, out_path: str, image_spec: dict) -> dict:
    import rosbag2_py
    from geometry_msgs.msg import PoseStamped
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image, JointState

    with open(os.path.join(bag_dir, "episode.json")) as f:
        meta = json.load(f)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    topics = {"/scene_camera/rgb": Image, "/wrist_camera/rgb": Image, "/joint_states": JointState,
              "/joint_commands": JointState, "/ee_pose": PoseStamped}
    topics |= {f"/blocks/{c}/pose": PoseStamped for c in BLOCK_COLORS}
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    images = {"/scene_camera/rgb": {}, "/wrist_camera/rgb": {}}
    states, commands, poses = [], [], []
    blocks = {c: [] for c in BLOCK_COLORS}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, topics[topic])
        t = round(stamp(msg), 4)
        if topic in images:
            rgb = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, -1)[..., :3]
            images[topic][t] = preprocess(rgb, image_spec[topic.split("_")[0].lstrip("/")])
        elif topic == "/ee_pose" or topic.startswith("/blocks/"):
            p, q = msg.pose.position, msg.pose.orientation
            series = poses if topic == "/ee_pose" else blocks[topic.split("/")[2]]
            series.append((t, [p.x, p.y, p.z, q.w, q.x, q.y, q.z]))
        else:
            index = [list(msg.name).index(n) for n in JOINT_NAMES]
            position = np.asarray(msg.position)[index]
            velocity = np.asarray(msg.velocity)[index] if len(msg.velocity) else np.zeros(len(JOINT_NAMES))
            (states if topic == "/joint_states" else commands).append((t, position, velocity))

    times = sorted(set(images["/scene_camera/rgb"]) & set(images["/wrist_camera/rgb"]))
    if not times or not states or not commands:
        raise RuntimeError("missing camera or joint data")

    def latest(series, t):
        keys = np.array([s[0] for s in series])
        return series[max(0, int(np.searchsorted(keys, t + 1e-6)) - 1)]

    # The commanded grasp-point goal: the target of the move_to being executed (or last executed) at each frame
    goal_changes = [(times[0] - 1.0, np.asarray(latest(poses, times[0])[1][:3]) - [0.0, 0.0, GRASP_OFFSET_Z])]
    for step in meta.get("steps", []):
        decision = step.get("decision") or {}
        if decision.get("action") == "move_to":
            result = step.get("result") or {}
            target = result.get("clipped_target") or [decision["x"], decision["y"], decision["z"]]
            goal_changes.append((step["sim_time"] - result.get("duration_s", 0.0), np.asarray(target)))
    goal_times = np.array([g[0] for g in goal_changes])
    goals = np.stack([goal_changes[max(0, int(np.searchsorted(goal_times, t + 1e-6)) - 1)][1] for t in times])

    np.savez(
        out_path,
        scene_rgb=np.stack([images["/scene_camera/rgb"][t] for t in times]),
        wrist_rgb=np.stack([images["/wrist_camera/rgb"][t] for t in times]),
        joint_positions=np.stack([latest(states, t)[1] for t in times]).astype(np.float32),
        joint_velocities=np.stack([latest(states, t)[2] for t in times]).astype(np.float32),
        actions=np.stack([latest(commands, t)[1] for t in times]).astype(np.float32),
        ee_pose=np.asarray([latest(poses, t)[1] for t in times], dtype=np.float32),
        block_poses=np.asarray([[latest(blocks[c], t)[1] for c in BLOCK_COLORS] for t in times], dtype=np.float32),
        gripper_command=np.asarray([latest(commands, t)[1][7] > 0.02 for t in times], dtype=np.float32),
        goal_position=goals.astype(np.float32),
        timestamps=np.asarray(times),
        task_index=meta["task"]["task_index"],
        success=meta["success"],
        seed=meta["seed"],
        image_spec=json.dumps(image_spec),
    )
    return {"episode": meta["episode"], "frames": len(times), "success": meta["success"],
            "task_index": meta["task"]["task_index"], "num_steps": meta["num_steps"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episodes_dir")
    parser.add_argument("dataset_dir")
    parser.add_argument("--size", type=int, default=128, help="Square image size for both cameras")
    parser.add_argument("--scene-size", help="WxH for the scene camera (overrides --size)")
    parser.add_argument("--wrist-size", help="WxH for the wrist camera (overrides --size)")
    parser.add_argument("--scene-crop", help="x0,y0,x1,y1 crop of the scene image (source pixels) before resizing")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    def size(value):
        return [int(v) for v in value.lower().split("x")] if value else [args.size, args.size]

    image_spec = {"scene": {"crop": [int(v) for v in args.scene_crop.split(",")] if args.scene_crop else None,
                            "size": size(args.scene_size)},
                  "wrist": {"crop": None, "size": size(args.wrist_size)}}
    os.makedirs(args.dataset_dir, exist_ok=True)
    jobs = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for name in sorted(os.listdir(args.episodes_dir)):
            bag_dir = os.path.join(args.episodes_dir, name)
            out_path = os.path.join(args.dataset_dir, name + ".npz")
            if not os.path.isfile(os.path.join(bag_dir, "episode.json")) or not os.path.isfile(
                    os.path.join(bag_dir, "metadata.yaml")):
                continue
            if os.path.exists(out_path):
                continue
            jobs[name] = pool.submit(convert, bag_dir, out_path, image_spec)
        for name, job in jobs.items():
            try:
                print(json.dumps(job.result()), flush=True)
            except Exception as exc:
                print(f"{name}: skipped ({exc})", file=sys.stderr, flush=True)
    index = []
    for name in sorted(os.listdir(args.dataset_dir)):
        if name.endswith(".npz"):
            with np.load(os.path.join(args.dataset_dir, name)) as data:
                index.append({"file": name, "frames": int(len(data["timestamps"])),
                              "success": bool(data["success"]), "task_index": int(data["task_index"])})
    with open(os.path.join(args.dataset_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(f"{len(index)} episodes, {sum(e['success'] for e in index)} successful, "
          f"{sum(e['frames'] for e in index)} frames")


if __name__ == "__main__":
    main()

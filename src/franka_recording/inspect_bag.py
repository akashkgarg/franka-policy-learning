"""Summarize a recorded Franka bag and dump one RGB frame per camera as PPM images.

Run with the system ROS 2 Python: python3.12 inspect_bag.py <bag_dir> [out_dir]
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

bag_dir = sys.argv[1]
out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else bag_dir)

reader = rosbag2_py.SequentialReader()
reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id="mcap"),
            rosbag2_py.ConverterOptions("cdr", "cdr"))
types = {t.name: t.type for t in reader.get_all_topics_and_types()}

tf_edges = set()
stamps = defaultdict(list)
samples = {}
max_contact = defaultdict(float)
dumped = set()
phases = []
while reader.has_next():
    topic, data, _ = reader.read_next()
    msg = deserialize_message(data, get_message(types[topic]))
    if hasattr(msg, "header"):
        stamps[topic].append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
    if topic in ("/tf", "/tf_static"):
        for t in msg.transforms:
            tf_edges.add((t.header.frame_id, t.child_frame_id, topic))
    elif topic.endswith("/contact"):
        f = msg.wrench.force
        max_contact[topic] = max(max_contact[topic], float(np.linalg.norm([f.x, f.y, f.z])))
    elif topic == "/pick_place/phase":
        if not phases or phases[-1] != msg.data:
            phases.append(msg.data)
    elif topic.endswith("/rgb") and topic not in dumped and len(stamps[topic]) == 60:
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3]
        path = out_dir / f"{topic.strip('/').replace('/', '_')}.ppm"
        with open(path, "wb") as f:
            f.write(f"P6 {msg.width} {msg.height} 255\n".encode() + image.tobytes())
        dumped.add(topic)
        print(f"wrote {path} ({msg.encoding})")
    if topic not in samples:
        samples[topic] = msg

print("\nTF edges:")
for parent, child, topic in sorted(tf_edges):
    print(f"  {topic}: {parent} -> {child}")
print("\nPhases:", " -> ".join(phases))
print("\nMax contact force [N]:", dict(max_contact))
js = samples.get("/joint_states")
if js:
    print("\nJoint names:", list(js.name))
for topic in sorted(stamps):
    s = stamps[topic]
    if len(s) > 1:
        print(f"{topic:45s} n={len(s):4d} t=[{s[0]:.3f}, {s[-1]:.3f}] rate={(len(s) - 1) / max(s[-1] - s[0], 1e-9):.1f} Hz")
for topic in ("/scene_camera/depth", "/scene_camera/points", "/scene_camera/semantic_labels", "/scene_camera/bbox_3d",
              "/hand_imu"):
    m = samples.get(topic)
    if m is None:
        continue
    if hasattr(m, "encoding"):
        print(f"{topic}: {m.width}x{m.height} {m.encoding}")
    elif hasattr(m, "point_step"):
        print(f"{topic}: {m.width * m.height} points, fields {[f.name for f in m.fields]}")
    elif hasattr(m, "detections"):
        print(f"{topic}: {len(m.detections)} detections, ids {[r.hypothesis.class_id for d in m.detections for r in d.results]}")
    elif hasattr(m, "data"):
        print(f"{topic}: {m.data[:200]}")
    else:
        print(f"{topic}: {m}")

"""Render the RGB camera topics of a recorded Franka bag into a side-by-side MP4 and GIF.

Run with the system ROS 2 Python: python3.12 bag_to_video.py <bag_dir> [out_prefix]
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

CAMERAS = ("/scene_camera/rgb", "/wrist_camera/rgb")
GIF_WIDTH = 640
GIF_FPS = 10

bag_dir = Path(sys.argv[1])
out_prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else bag_dir / "franka_pick_place"

reader = rosbag2_py.SequentialReader()
reader.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"),
            rosbag2_py.ConverterOptions("cdr", "cdr"))
reader.set_filter(rosbag2_py.StorageFilter(topics=list(CAMERAS)))

frames = {topic: {} for topic in CAMERAS}
while reader.has_next():
    topic, data, _ = reader.read_next()
    msg = deserialize_message(data, Image)
    stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
    frames[topic][stamp] = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3]

stamps = sorted(set.intersection(*(set(f) for f in frames.values())))
if len(stamps) < 2:
    sys.exit(f"not enough synchronized frames in {bag_dir}")
fps = round((len(stamps) - 1) / ((stamps[-1] - stamps[0]) * 1e-9))
height, width = np.hstack([frames[t][stamps[0]] for t in CAMERAS]).shape[:2]

mp4 = out_prefix.with_suffix(".mp4")
ffmpeg = subprocess.Popen(
    ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
     "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
     str(mp4)],
    stdin=subprocess.PIPE)
for stamp in stamps:
    ffmpeg.stdin.write(np.ascontiguousarray(np.hstack([frames[t][stamp] for t in CAMERAS])).tobytes())
ffmpeg.stdin.close()
if ffmpeg.wait() != 0:
    sys.exit("ffmpeg failed writing the MP4")

gif = out_prefix.with_suffix(".gif")
subprocess.run(
    ["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp4), "-vf",
     f"fps={GIF_FPS},scale={GIF_WIDTH}:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];"
     "[b][p]paletteuse=dither=bayer:bayer_scale=4", "-loop", "0", str(gif)],
    check=True)
print(f"{len(stamps)} frames at {fps} fps ({width}x{height})\nwrote {mp4}\nwrote {gif}")

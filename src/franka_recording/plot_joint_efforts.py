"""Plot Franka joint efforts and fingertip contact forces from a recorded bag.

Run with ROS 2 Jazzy sourced:
    python3.12 plot_joint_efforts.py <bag_dir> [output.png]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
from geometry_msgs.msg import WrenchStamped
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState
from std_msgs.msg import String

TOPICS = {
    "/joint_states": JointState,
    "/left_finger/contact": WrenchStamped,
    "/right_finger/contact": WrenchStamped,
    "/pick_place/phase": String,
}


def stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def read_bag(bag: str):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(TOPICS)))
    names, joint_t, efforts = [], [], []
    contacts = {"/left_finger/contact": ([], []), "/right_finger/contact": ([], [])}
    phases = []
    last_joint_t = 0.0
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, TOPICS[topic])
        if topic == "/joint_states":
            names = list(msg.name)
            last_joint_t = stamp(msg)
            joint_t.append(last_joint_t)
            efforts.append(msg.effort)
        elif topic == "/pick_place/phase":
            if not phases or phases[-1][1] != msg.data:
                phases.append((last_joint_t, msg.data))
        else:
            f = msg.wrench.force
            contacts[topic][0].append(stamp(msg))
            contacts[topic][1].append(np.linalg.norm([f.x, f.y, f.z]))
    return names, np.array(joint_t), np.array(efforts), contacts, phases


def shade_phases(ax, phases, t_end: float, label: bool) -> None:
    colors = plt.cm.Pastel1(np.linspace(0, 1, 9))
    for i, (start, name) in enumerate(phases):
        end = phases[i + 1][0] if i + 1 < len(phases) else t_end
        ax.axvspan(start, end, color=colors[i % len(colors)], alpha=0.5, lw=0)
        if label:
            ax.text((start + end) / 2, 1.02, name, transform=ax.get_xaxis_transform(),
                    rotation=35, ha="left", va="bottom", fontsize=7)


def main() -> None:
    bag = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(bag) / "joint_efforts.png"
    names, t, efforts, contacts, phases = read_bag(bag)
    arm = [i for i, n in enumerate(names) if "finger" not in n]
    fingers = [i for i, n in enumerate(names) if "finger" in n]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    for ax in axes:
        shade_phases(ax, phases, t[-1], label=ax is axes[0])
        ax.grid(alpha=0.3)

    for i in arm:
        axes[0].plot(t, efforts[:, i], lw=1.2, label=names[i])
    axes[0].set_ylabel("Arm joint torque [N·m]")
    axes[0].legend(ncol=7, fontsize=8, loc="lower left")

    for i in fingers:
        axes[1].plot(t, efforts[:, i], lw=1.2, label=names[i])
    axes[1].set_ylabel("Finger joint force [N]")
    axes[1].legend(fontsize=8, loc="upper left")

    for topic, (ct, cf) in contacts.items():
        axes[2].plot(ct, cf, lw=1.2, label=topic.split("/")[1])
    axes[2].set_ylabel("Fingertip contact |F| [N]")
    axes[2].set_xlabel("Sim time [s]")
    axes[2].legend(fontsize=8, loc="upper left")

    fig.suptitle(f"Franka joint efforts — {Path(bag).name}", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

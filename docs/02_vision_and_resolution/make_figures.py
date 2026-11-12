"""Build the figures for README.md from the eval results and recorded images."""

import glob
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

DATA = Path.home() / "franka_llm_data"
OUT = Path(__file__).parent / "assets"
OUT.mkdir(exist_ok=True)
MODES = ["success", "near miss", "stacked on the wrong block", "dropped elsewhere", "fumbled the grasp",
         "never picked", "picked the wrong block"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def successes(eval_dir):
    s = json.load(open(DATA / eval_dir / "summary.json"))
    return s["successes"], s["episodes"]


def failure_modes(eval_dir):
    counts = dict.fromkeys(MODES, 0)
    for f in glob.glob(str(DATA / eval_dir / "policy_seed*/episode.json")):
        e = json.load(open(f))
        top, bottom = e["task"]["top"], e["task"]["bottom"]
        end = e["block_positions"][top]
        moved = math.dist(e["initial_block_positions"][top], end)
        off_bottom = math.dist(end[:2], e["block_positions"][bottom][:2])
        if e["success"]:
            mode = "success"
        elif moved < 0.02:
            mode = "picked the wrong block" if max(e["displacement_of_other_blocks_m"].values()) > 0.02 else "never picked"
        elif end[2] > 0.04:
            mode = "stacked on the wrong block" if off_bottom > 0.04 else "near miss"
        elif moved < 0.08:
            mode = "fumbled the grasp"
        elif off_bottom < 0.10:
            mode = "near miss"
        else:
            mode = "dropped elsewhere"
        counts[mode] += 1
    return counts


def bar(ax, labels, values, colors, total=70):
    bars = ax.bar(labels, values, color=colors)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v}/{total}", ha="center", fontsize=10)
    ax.set_ylim(0, total)


# State vs vision, on both tables.
fig, ax = plt.subplots(figsize=(9, 4))
rows = [("goal_state\n(knows block xyz)", "eval_cmp_goal_state_grid", "eval_cmp_goal_state_wood"),
        ("goal_vision\n(grid LLM data)", "eval_cmp_goal_vision_grid", "eval_cmp_goal_vision_wood"),
        ("wood_vision\n(vision-only LLM data)", "eval_cmp_wood_vision_grid", "eval_wood_vision")]
x = range(len(rows))
for offset, col, color, name in ((-0.2, 1, "#8a8f98", "grid table"), (0.2, 2, "#b5793b", "wood table")):
    vals = [successes(r[col])[0] for r in rows]
    bars = ax.bar([i + offset for i in x], vals, 0.4, color=color, label=name)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, str(v), ha="center", fontsize=10)
ax.set_xticks(list(x), [r[0] for r in rows])
ax.set_ylabel("successes out of 70")
ax.set_ylim(0, 70)
ax.legend(frameon=False)
ax.set_title("Every policy falls apart on the table it wasn't trained on")
fig.tight_layout()
fig.savefig(OUT / "state_vs_vision.png", dpi=130)

# Resolution sweep + longer training.
runs = [("128x128\n(squashed)", "eval_wood_vision"), ("crop\n264x176", "eval_wood_crop264"),
        ("256x192", "eval_wood_res256"), ("320x240", "eval_wood_res320"), ("640x480", "eval_wood_res640"),
        ("256x192\n40k steps", "eval_wood_res256_40k"), ("256x192\n80k steps", "eval_wood_res256_80k")]
fig, ax = plt.subplots(figsize=(10, 4))
colors = ["#9aa5b1"] * 5 + ["#3b7bb5"] * 2
colors[2] = "#3b7bb5"
bar(ax, [r[0] for r in runs], [successes(r[1])[0] for r in runs], colors)
ax.axvline(4.5, color="#ccc", ls="--")
ax.text(2, 64, "image resolution (all 20k steps)", ha="center", color="#666")
ax.text(5.5, 64, "train longer", ha="center", color="#666")
ax.set_ylabel("successes out of 70 (wood)")
fig.tight_layout()
fig.savefig(OUT / "resolution_and_steps.png", dpi=130)

# Failure modes.
modes = MODES
mode_colors = ["#4caf50", "#a6d96a", "#3b7bb5", "#f08a4b", "#f2c14e", "#d1495b", "#7d5ba6"]
fig, ax = plt.subplots(figsize=(10, 4.6))
bottom = [0] * len(runs)
counts = [failure_modes(r[1]) for r in runs]
for mode, color in zip(modes, mode_colors):
    vals = [c[mode] for c in counts]
    ax.bar([r[0] for r in runs], vals, bottom=bottom, color=color, label=mode)
    bottom = [b + v for b, v in zip(bottom, vals)]
ax.set_ylabel("episodes (of 70)")
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "failure_modes.png", dpi=130)
json.dump({r[1]: c for r, c in zip(runs, counts)}, open(OUT / "failure_modes.json", "w"), indent=1)


def row(paths, out, height=300, gap=8):
    ims = [Image.open(p).convert("RGB") for p in paths]
    ims = [im.resize((round(im.width * height / im.height), height), Image.NEAREST) for im in ims]
    canvas = Image.new("RGB", (sum(i.width for i in ims) + gap * (len(ims) - 1), height), "white")
    xpos = 0
    for im in ims:
        canvas.paste(im, (xpos, 0))
        xpos += im.width + gap
    canvas.save(OUT / out, quality=90)


grid_ep = DATA / "vision/grid_episodes/vision_seed100000/llm_images"
wood_ep = DATA / "vision/wood_episodes/vision_seed100000/llm_images"
row([grid_ep / "step00_scene.jpg", wood_ep / "step00_scene.jpg"], "tables_grid_vs_wood.jpg")
row([wood_ep / "step00_scene.jpg", wood_ep / "step00_wrist.jpg"], "llm_view_start.jpg")
row([wood_ep / "step08_scene.jpg", wood_ep / "step08_wrist.jpg"], "llm_view_place.jpg")

# What the policy actually gets to see at each resolution (upscaled with nearest so the pixels show).
raw = Image.open("/tmp/raw_scene_camera.png").convert("RGB")
variants = []
for name, size in (("128", (128, 128)), ("256", (256, 192)), ("640", (640, 480))):
    p = f"/tmp/res_{name}.png"
    raw.resize(size, Image.BILINEAR).resize((640, 480), Image.NEAREST).crop((240, 120, 480, 300)).save(p)
    variants.append(p)
row(variants, "resolution_examples.jpg", height=280)
print("done")

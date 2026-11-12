# Franka (Panda) policy learning in Isaac Sim

Prototyping work from Oct 2026 on teaching a simulated Franka Panda to stack blocks. An LLM (`gpt-6-astra`
via the LLM Gateway) drives the arm through a small set of high-level actions executed by RMPflow; every
episode is recorded to ROS 2 MCAP bags, and an ACT-style imitation policy is trained on those recordings
to replace the LLM.

Headline results:

- **LLM teacher:** 607/607 stacks with full world state, and near-perfect from camera images alone using a
  `locate(camera, u, v)` depth-deprojection tool.
- **Distilled policy (state):** 67% (14/21) when predicting the LLM's waypoint + gripper state; predicting
  next hand position or joint targets got 0%.
- **Distilled policy (vision only):** about 40% (29/70) after the resolution sweep, trained on wood-table data
  at 256×192 for 80k steps.

## Reports

| report | what it covers |
|---|---|
| [`docs/01_llm_teacher_and_distillation/REPORT.md`](docs/01_llm_teacher_and_distillation/REPORT.md) | Isaac Sim streaming on H200s, MCAP recording, LLM-driven collection, first policy experiments, the Franka LLM Lab web app |
| [`docs/02_vision_and_resolution/README.md`](docs/02_vision_and_resolution/README.md) | Vision-only LLM teacher, grid vs. wood table, image resolution / crop / training-length sweep, failure modes |

Both reports were written against the original layout (`~/isaacsim/tools/...`, `~/franka_llm_data/...`);
see the path mapping below.

## Layout

```
src/franka_recording/   collection, dataset conversion, training and eval
  franka_pick_place_ros2.py, record_franka_bag.sh   scripted single-cube demo -> MCAP bag
  franka_llm_episodes.py                            LLM / policy-driven stacking episodes (sim worker)
  collect_llm_episodes.sh                           one worker per GPU for LLM data collection
  bags_to_dataset.py                                MCAP bags -> per-episode npz at 15 Hz
  train_policy.py                                   ACT-style policy (ResNet-18 per camera + transformer)
  eval_policy.sh, eval_when_trained.sh              held-out seed evaluation
  vision_vs_state_pipeline.sh, resolution_sweep_eval.sh   end-to-end experiment pipelines
  inspect_bag.py, bag_to_video.py, plot_joint_efforts.py  bag utilities
src/franka_lab/         "Franka LLM Lab" web app: episode browser, dataset overview, live run console
src/dev_scripts/        probes (NVENC, Franka asset) and Playwright stream/lab screenshot checks
launchers/              patched isaac-sim.streaming.sh / isaac-sim.webviewer.sh (Vulkan ICD fixes, software encoder on GPUs without NVENC)
docs/                   reports, figures, videos, and the figure-generation script
results/                eval summaries per policy, sweep comparisons, training logs (jsonl)
requirements/policy.txt frozen training venv (Python 3.12, torch 2.14 + cu130)
```

Not included (too large; still on the h200-training devspace under `~/franka_llm_data/`): the 133 GB of
LLM episode bags, the npz datasets, policy checkpoints (~120 MB each), and per-episode eval bags.

## Setup

Requirements: Isaac Sim 6.1 (standalone), ROS 2 Jazzy at `/opt/ros/jazzy`, and a Python 3.12 venv for training.

```bash
python3.12 -m venv ~/.venvs/franka-policy
~/.venvs/franka-policy/bin/pip install -r requirements/policy.txt

# scripts locate Isaac Sim two directories above themselves; either symlink into the install...
ln -s "$PWD/src/franka_recording" ~/isaacsim/tools/franka_recording
ln -s "$PWD/src/franka_lab"       ~/isaacsim/tools/franka_lab
# ...or point at it explicitly
export ISAAC_SIM_DIR=~/isaacsim
```

LLM runs need an LCA token at `~/.config/llm-gateway/lca_token` (or pasted into the Lab's Run tab). Policy
runs don't.

## Running

```bash
# scripted single-cube pick-and-place recorded to MCAP
src/franka_recording/record_franka_bag.sh

# LLM data collection, one worker per GPU
src/franka_recording/collect_llm_episodes.sh

# bags -> dataset -> policy -> eval
~/.venvs/franka-policy/bin/python src/franka_recording/bags_to_dataset.py --help
~/.venvs/franka-policy/bin/python src/franka_recording/train_policy.py --help
src/franka_recording/eval_policy.sh

# web app
src/franka_lab/run.sh --port 8240 --gpu 1 --data ~/franka_llm_data
```

Gotchas from the original work: ROS 2 Jazzy needs `python3.12` inside `bash` (sourcing `setup.bash`), give
each sim worker its own `ROS_DOMAIN_ID` with `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`, and H200s have no
NVENC so streaming needs the software encoder (the patched launcher handles that).

## Path mapping from the original workspace

| original | here |
|---|---|
| `~/isaacsim/tools/franka_recording/` | `src/franka_recording/` |
| `~/isaacsim/tools/franka_lab/` | `src/franka_lab/` |
| `~/isaacsim/tools/dev_scripts/` | `src/dev_scripts/` |
| `~/isaacsim/isaac-sim.{streaming,webviewer}.sh` | `launchers/` |
| `~/franka_report/` | `docs/01_llm_teacher_and_distillation/` |
| `~/franka_llm_data/report/` | `docs/02_vision_and_resolution/` |
| `~/franka_llm_data/eval_<name>/summary.json` | `results/eval/eval_<name>.json` |
| `~/franka_llm_data/runs/<name>/train_log.jsonl` | `results/train_logs/<name>.jsonl` |

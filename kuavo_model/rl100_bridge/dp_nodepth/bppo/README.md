# RL-100 Offline BPPO for LeTools DP No-Depth

This package connects an RGB-only LeTools `DiffusionPolicy` checkpoint to the
offline BPPO training components in RL-100. The policy adapter, transition
alignment, Critic/Value training, ensemble Dynamics training, BPPO update, and
LeTools checkpoint export are contained in this package.

## Expert-only BPPO

The currently supported task adapter is `icra_task1`. For successful
demonstrations it assigns `reward=1, done=1` at the later stable left/right
gripper-release frame. All earlier transitions have zero reward, and samples
after that terminal frame are excluded. Dataset-specific exclusions
and gripper indices are configured in
`configs/train/rl100_dp_nodepth/icra_task1/offline_bppo.yaml`.

RL-100 is expected as a Git submodule at `third_party/rl100`, pinned to commit
`64264d952c1fda9d5096c090ddaa7177757a77ad`. Its Python source root is
`third_party/rl100/RL-100`.

Generate the task-specific terminal labels once from the LeRobot dataset:

```bash
python -m \
  kuavo_model.rl100_bridge.dp_nodepth.bppo.tasks.icra_task1.release_terminal \
  --dataset-root /path/to/lerobot_dataset \
  --output /path/to/icra_task1_release_terminal.csv \
  --exclude-episode 579 \
  --exclusion-reason "converted episode is truncated before right release"
```

This creates both the CSV and the adjacent `.exclusions.json` audit required
by the training loader. Episode 579 is the one known truncated conversion in
the validated 1000-demonstration dataset; this exclusion is data-version
specific and must be reviewed if another dataset is used.

For the expert-only route, keep `dynamics.terminal_mode: reward_threshold`
in `configs/train/rl100_dp_nodepth/icra_task1/offline_bppo.yaml`. This uses
the original reward-based terminal handling and does not require flywheel
metadata or the learned terminal classifier.

After replacing every `/path/to/...` entry in the configuration, run a
non-training preflight with:

```bash
python kuavo_model/train_rl100_bppo_dp_nodepth.py \
  --config configs/train/rl100_dp_nodepth/icra_task1/offline_bppo.yaml \
  --check-only
```

Start the complete Critic, Dynamics, and 8000-step BPPO pipeline with:

```bash
python kuavo_model/train_rl100_bppo_dp_nodepth.py \
  --config configs/train/rl100_dp_nodepth/icra_task1/offline_bppo.yaml
```

The exported `checkpoints/bppo_*/pretrained_model` directories are standard
LeTools/LeRobot Diffusion Policy checkpoint bundles.

## ICRA Task 1 simulation flywheel

For simulation collection, conversion, and mixed-outcome annotations, see the
[flywheel guide](../../../../kuavo_data/flywheel/README.md).

`configs/train/rl100_dp_nodepth/icra_task1/offline_bppo_flywheel.yaml` is a
starting template for mixed outcomes; set paths and `expected_episode_count`
to match the actual dataset. The template's value of 50 is an example. Its
`num_inference_steps: 10` follows the original BPPO configuration; align it
with the selected BC checkpoint and the intended experiment.
Its learned terminal classifier is experimental. It predicts terminal status
from state and action without elapsed time, so it cannot reliably distinguish
a time-limit ending from the same state at another time. A minimal smoke test
does not establish classifier accuracy, policy improvement, or deployment
readiness; evaluate on representative held-out outcomes before using it.

# ICRA Task 1 Simulation Flywheel

This workflow collects simulation rosbag data for the existing LeTools data
conversion and DP no-depth offline BPPO training pipeline. The current
collector targets ICRA Task 1 with rq2f85 grippers; real-robot collection is
not covered by this implementation. Paths below are examples and must be
replaced for the local environment.

## Collect and inspect

Start the ICRA simulator, then run the collector with an existing simulation
deployment configuration and a local copy of the recording template
`configs/deploy/flywheel/icra_task1_sim_flywheel.yaml`. Set its `output_root`
to a writable location outside the repository.

```bash
python kuavo_deploy/src/scripts/script_auto_test_flywheel.py \
  --config /path/to/simulation_deploy.yaml \
  --flywheel-config /path/to/local_flywheel_record.yaml
```

Each collected episode has a rosbag and metadata. Inspect the results with:

```bash
python kuavo_data/flywheel/inventory.py /path/to/collection_run
```

The inventory reports `success`, `timeout`, or `invalid` based on metadata.
Review timeout outcomes before treating them as failures; invalid recordings
must not be treated as failed task attempts. Use successful trajectories for
BC training. This inventory does not independently inspect the simulator score.

## Convert and map

Convert the selected expert and flywheel bags using the existing
`kuavo_data/CvtRosbag2Lerobot.py` and retain the complete conversion log.
Prepare a source CSV with columns `bag_path,source,metadata_path`. Set `source`
to `flywheel` or `expert_success`; expert rows leave `metadata_path` empty
and must correspond to known successful demonstrations. Bag paths in the CSV
must match those recorded by the converter.

```bash
python kuavo_data/flywheel/conversion_map.py \
  --conversion-log /path/to/conversion.log \
  --sources /path/to/sources.csv \
  --dataset-root /path/to/lerobot_dataset \
  --output /path/to/episode_mapping.csv
```

The mapping tool checks the conversion log and per-episode frame counts.
Do not infer episode indices from bag names: conversion may reorder or skip
input bags. Check message timing and action alignment before using collected
data for training.

## Create mixed reward labels

```bash
python -m kuavo_model.rl100_bridge.dp_nodepth.bppo.tasks.icra_task1.outcomes_flywheel \
  --dataset-root /path/to/lerobot_dataset \
  --mapping /path/to/episode_mapping.csv \
  --output /path/to/reward_annotations.csv \
  --timeout-as-terminal
```

Successful trajectories receive reward 1 at the later stable gripper release.
The explicit flag treats a reviewed timeout as a fixed-horizon terminal at the
final converted frame, with reward 0 throughout. This is a modeling choice:
a truncated trajectory is not necessarily a terminal state. The command also
creates the adjacent `.exclusions.json` audit required by the BPPO loader.

For BC, train on successful data only. For mixed-outcome BPPO, use the
`configs/train/rl100_dp_nodepth/icra_task1/offline_bppo_flywheel.yaml` template
and set its paths and episode count for the selected dataset. See the
[BPPO guide](../../kuavo_model/rl100_bridge/dp_nodepth/bppo/README.md) for
training and the learned terminal classifier's limitations.

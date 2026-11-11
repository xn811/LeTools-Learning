"""Collect ICRA Task 1 simulation rollouts."""

import argparse
from pathlib import Path
import signal


def main():
    parser = argparse.ArgumentParser(description="Collect ICRA Task 1 simulation rosbags")
    parser.add_argument("--config", required=True, help="Existing simulation deploy config")
    parser.add_argument("--flywheel-config", required=True, help="Recording config")
    args = parser.parse_args()

    import rospy
    import yaml
    from kuavo_deploy.config import load_kuavo_config
    from kuavo_deploy.src.eval.sim_auto_test_flywheel import collect_flywheel

    config = load_kuavo_config(args.config)
    with Path(args.flywheel_config).open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream) or {}
    if config.env.inference_env != "sim" or config.env.real:
        raise ValueError("Flywheel collection is restricted to the simulation environment")
    rospy.init_node("kuavo_flywheel_collect", anonymous=True)
    # rospy's default SIGINT handler only requests shutdown. Some existing
    # environment initializers do not check that flag, so interrupt the main
    # thread and let the flywheel cleanup code run instead.
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt("Interrupted by user")

    signal.signal(signal.SIGINT, interrupt)
    collect_flywheel(config, settings)


if __name__ == "__main__":
    main()

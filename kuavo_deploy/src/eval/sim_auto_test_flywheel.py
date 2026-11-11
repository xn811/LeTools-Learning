"""Record ICRA simulation episodes for the existing rosbag conversion pipeline.

Bag topic presence is only a preliminary check. Conversion, action semantics,
time alignment and release annotation must still be audited before training.
"""

import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import traceback
import uuid
from contextlib import contextmanager

import gymnasium as gym
import rosbag
import rosgraph
import rospy
import rosservice
import torch
from lerobot.utils.random_utils import set_seed
from std_msgs.msg import Bool
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

from kuavo_deploy.src.eval import sim_auto_test as original
from kuavo_deploy.utils.obs_buffer import CompressedImage, JointState, sensorsData
from kuavo_deploy.utils.policy_loader import inject_task_prompt


# Use the exact same message classes as the deployment observation buffer.
# rospy shares subscriptions to a topic within a process: AnyMsg here can
# cause the existing typed observation callbacks to receive AnyMsg objects.
_OBSERVATION_MESSAGE_TYPES = {
    "/sensors_data_raw": sensorsData,
    "/gripper/state": JointState,
    "/cam_h/color/image_raw/compressed": CompressedImage,
    "/cam_l/color/image_raw/compressed": CompressedImage,
    "/cam_r/color/image_raw/compressed": CompressedImage,
    "/cam_h/depth/image_raw/compressedDepth": CompressedImage,
    "/cam_l/depth/image_rect_raw/compressedDepth": CompressedImage,
    "/cam_r/depth/image_rect_raw/compressedDepth": CompressedImage,
}


def _write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


@contextmanager
def _bounded_initialization(seconds):
    """Interrupt Python-level blocking initialization in the main thread.

    SIGALRM is Unix-specific and cannot reliably preempt a stuck native call.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Flywheel initialization must run in the main thread")
    previous = signal.getsignal(signal.SIGALRM)
    if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
        raise RuntimeError("Cannot use initialization timeout while another alarm is active")

    def timeout_handler(_signum, _frame):
        raise TimeoutError(f"Environment initialization exceeded {seconds:g} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _service_uri(name):
    try:
        return rosservice.get_service_uri(name)
    except Exception:
        return None


class _InitGate:
    """Accept init only from the new /simulator/reset service generation."""

    def __init__(self):
        self.lock = threading.Lock()
        self.previous_uri = None
        self.accepted_uri = None
        self.armed = False

    def arm(self, previous_uri):
        if not previous_uri:
            raise RuntimeError("Cannot identify the previous simulator reset service")
        with self.lock:
            self.previous_uri = previous_uri
            self.accepted_uri = None
            self.armed = True

    def callback(self, request):
        # Master lookup happens at callback time, not later, so an old init
        # cannot be retroactively associated with a new process.
        caller = (getattr(request, "_connection_header", None) or {}).get("callerid")
        uri = _service_uri("/simulator/reset")
        with self.lock:
            valid = (self.armed and caller == "/simulator_task1" and uri
                     and uri != self.previous_uri)
            if valid:
                self.accepted_uri = uri
        if valid:
            return TriggerResponse(success=True, message="New simulator generation accepted")
        rospy.logwarn("Ignored stale or unidentifiable simulator init: caller=%s uri=%s", caller, uri)
        return TriggerResponse(success=False, message="Stale or unidentifiable simulator generation")

    def current(self):
        with self.lock:
            return self.accepted_uri

    def disarm(self):
        with self.lock:
            self.armed = False


def _wait_init(gate, timeout):
    deadline = time.monotonic() + timeout
    while True:
        if rospy.is_shutdown() or not original.check_control_signals():
            raise InterruptedError("Collection stopped while waiting for simulator init")
        accepted = gate.current()
        if accepted and _service_uri("/simulator/reset") == accepted:
            gate.disarm()
            return accepted
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for init from the new simulator generation")
        time.sleep(0.1)


def _reset_simulator(reset, gate):
    previous_uri = _service_uri("/simulator/reset")
    gate.arm(previous_uri)
    response = reset(TriggerRequest())
    if not response.success:
        raise RuntimeError(f"Simulator reset failed: {response.message}")


class _FreshnessMonitor:
    """Measure wall-clock receipt ages; ROS header and policy buffer ages differ."""

    def __init__(self, topics, max_age):
        self.topics = topics
        self.max_age = max_age
        self.lock = threading.Lock()
        self.last_received = {}
        self.subscribers = []

    def start(self):
        for topic in self.topics:
            message_type = _OBSERVATION_MESSAGE_TYPES.get(topic)
            if message_type is None:
                raise ValueError(f"No deployment-compatible message type for observation topic: {topic}")
            self.subscribers.append(rospy.Subscriber(
                topic, message_type, self._received, callback_args=topic, queue_size=1
            ))

    def _received(self, _message, topic):
        with self.lock:
            self.last_received[topic] = time.monotonic()

    def stale(self):
        now = time.monotonic()
        with self.lock:
            return {topic: None if topic not in self.last_received else
                    round(now - self.last_received[topic], 3)
                    for topic in self.topics
                    if topic not in self.last_received or
                    now - self.last_received[topic] > self.max_age}

    def wait_ready(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if rospy.is_shutdown():
                raise InterruptedError("ROS shut down while waiting for observations")
            stale = self.stale()
            if not stale:
                return
            time.sleep(0.05)
        raise RuntimeError(f"Observation topics not ready: {self.stale()}")

    def close(self):
        for subscriber in self.subscribers:
            subscriber.unregister()


class _Recorder:
    def __init__(self, bag_path, topics, ready_timeout, shutdown_timeout):
        self.bag_path = bag_path
        self.topics = topics
        self.ready_timeout = ready_timeout
        self.shutdown_timeout = shutdown_timeout
        self.node_name = "/flywheel_record_" + uuid.uuid4().hex[:12]
        self.process = None

    def start(self):
        command = ["rosbag", "record", "-O", str(self.bag_path),
                   "__name:=" + self.node_name.lstrip("/")] + self.topics
        self.process = subprocess.Popen(command, start_new_session=True)
        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"rosbag recorder exited: {self.process.returncode}")
            # rosbag subscribes even if a publisher has not started yet. This
            # only establishes subscriber readiness, not message availability.
            _, subscribers, _ = rosgraph.Master(rospy.get_name()).getSystemState()
            registered = {topic for topic, nodes in subscribers if self.node_name in nodes}
            if set(self.topics) <= registered:
                return
            time.sleep(0.1)
        raise TimeoutError("rosbag did not register all requested subscriptions")

    def stop(self):
        if self.process is None:
            return None
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGINT)
        try:
            return self.process.wait(timeout=self.shutdown_timeout)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
            raise RuntimeError("rosbag failed to close cleanly")


def _inspect_bag(path, required, expected_types):
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("Finalized rosbag is missing or empty")
    with rosbag.Bag(str(path), "r") as bag:
        info = bag.get_type_and_topic_info().topics
        counts = {topic: int(value.message_count) for topic, value in info.items()}
        types = {topic: value.msg_type for topic, value in info.items()}
        missing = [topic for topic in required if counts.get(topic, 0) == 0]
        mismatches = {topic: {"expected": expected, "actual": types.get(topic)}
                      for topic, expected in expected_types.items()
                      if topic in types and types[topic] != expected}
        return {"counts": counts, "missing_required_topics": missing,
                "types": types, "type_mismatches": mismatches,
                "start_time": bag.get_start_time(), "end_time": bag.get_end_time()}


def _episode(config, policy, preprocessor, postprocessor, recorder, tail_seconds,
             generation_uri, observation_topics, max_observation_age, observation_ready_timeout,
             environment_init_timeout):
    cfg = config.inference
    result = {"success": False, "stop_reason": "unknown", "steps": 0,
              "success_ros_time": None, "success_wall_time": None}
    env = None
    subscriber = None
    monitor = _FreshnessMonitor(observation_topics, max_observation_age)
    last_generation_check = [0.0]

    def check_health():
        stale = monitor.stale()
        if stale:
            result["stale_observation_topics_seconds"] = stale
            raise RuntimeError(f"Observation messages stopped: {stale}")
        if recorder.process is not None and recorder.process.poll() is not None:
            raise RuntimeError(f"rosbag exited during episode: {recorder.process.returncode}")
        if rospy.is_shutdown():
            raise InterruptedError("ROS shutdown")
        if time.monotonic() - last_generation_check[0] >= 0.5:
            if _service_uri("/simulator/reset") != generation_uri:
                raise RuntimeError("Simulator service generation disappeared or changed")
            last_generation_check[0] = time.monotonic()

    def on_success(message):
        if message.data and result["success_ros_time"] is None:
            result["success"] = True
            result["success_ros_time"] = rospy.Time.now().to_sec()
            result["success_wall_time"] = time.time()

    try:
        monitor.start()
        monitor.wait_ready(observation_ready_timeout)
        if _service_uri("/simulator/reset") != generation_uri:
            raise RuntimeError("Simulator generation changed before episode start")
        with _bounded_initialization(environment_init_timeout):
            env = gym.make(config.env.env_name, max_episode_steps=cfg.max_episode_steps, config=config)
            subscriber = rospy.Subscriber("/simulator/success", Bool, on_success, queue_size=10)
            policy.reset()
            observation, _ = env.reset(seed=cfg.seed)
        check_health()
        recorder.start()
        check_health()
        if _service_uri("/simulator/reset") != generation_uri:
            raise RuntimeError("Simulator generation changed before /simulator/start")
        response = rospy.ServiceProxy("/simulator/start", Trigger)(TriggerRequest())
        if not response.success:
            raise RuntimeError(f"Simulator start failed: {response.message}")
        while not rospy.is_shutdown():
            if not original.check_control_signals():
                result["stop_reason"] = "interrupted"
                break
            check_health()
            if result["success"]:
                result["stop_reason"] = "success"
                break
            if cfg.policy_type != "client":
                observation = inject_task_prompt(observation, cfg.task_prompt)
            inputs = preprocessor(observation)
            with torch.inference_mode():
                action = postprocessor(policy.select_action(inputs))
            check_health()
            if result["success"]:
                result["stop_reason"] = "success"
                break
            observation, _, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
            result["steps"] += 1
            check_health()
            if result["success"]:
                result["stop_reason"] = "success"
                break
            if terminated or truncated:
                result["stop_reason"] = "terminated" if terminated else "time_limit"
                break
        if result["success"] and tail_seconds:
            deadline = time.monotonic() + tail_seconds
            while time.monotonic() < deadline:
                check_health()
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        if result["stop_reason"] == "unknown":
            raise InterruptedError("ROS shut down during episode")
    except BaseException as exc:
        result["stop_reason"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        try:
            result["rosbag_exit_code"] = recorder.stop()
        except Exception as exc:
            result["recording_error"] = str(exc)
        if subscriber is not None:
            subscriber.unregister()
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        monitor.close()
    return result


def collect_flywheel(config, settings):
    record = settings.get("record", {})
    topics = record.get("topics", [])
    required = record.get("required_topics", [])
    expected_types = record.get("expected_types", {})
    if not topics or len(set(topics)) != len(topics) or not set(required) <= set(topics):
        raise ValueError("Recording topics must be unique and include all required topics")
    if not isinstance(expected_types, dict) or not set(expected_types) <= set(topics):
        raise ValueError("expected_types must map only recorded topics to ROS types")
    root_value = settings.get("output_root", "")
    if not root_value or "/path/to/" in root_value:
        raise ValueError("Set output_root in a local flywheel configuration")
    tail = float(record.get("post_success_seconds", 3.0))
    ready = float(record.get("ready_timeout_seconds", 15))
    shutdown = float(record.get("shutdown_timeout_seconds", 30))
    init_timeout = float(record.get("init_timeout_seconds", 120))
    observation_topics = record.get("observation_topics", [
        "/sensors_data_raw", "/gripper/state",
        "/cam_h/color/image_raw/compressed", "/cam_l/color/image_raw/compressed",
        "/cam_r/color/image_raw/compressed",
    ])
    max_observation_age = float(record.get("max_observation_age_seconds", 5))
    observation_ready_timeout = float(record.get("observation_ready_timeout_seconds", 15))
    environment_init_timeout = float(record.get("environment_init_timeout_seconds", 30))
    if (not observation_topics or len(set(observation_topics)) != len(observation_topics)
            or not set(observation_topics) <= set(topics)):
        raise ValueError("observation_topics must be unique and included in recording topics")
    if tail < 0 or min(ready, shutdown, init_timeout, max_observation_age,
                       observation_ready_timeout, environment_init_timeout) <= 0:
        raise ValueError("Recording and init timeouts must be positive; tail must be nonnegative")
    root = Path(root_value).expanduser().resolve()
    run = root / (datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    bags = run / "bags"
    metadata = run / "metadata"
    bags.mkdir(parents=True, exist_ok=False)
    metadata.mkdir()
    cfg = config.inference
    set_seed(cfg.seed)
    policy, preprocessor, postprocessor, model_dir = original.setup_policy(
        Path(cfg.pretrained_path), cfg.policy_type, torch.device(cfg.device), task_prompt=cfg.task_prompt
    )
    gate = _InitGate()
    service = rospy.Service("/simulator/init", Trigger, gate.callback)
    reset = rospy.ServiceProxy("/simulator/reset", Trigger)
    results = []
    collection_error = None
    final_reset = {"attempted": False, "success": False}
    try:
        rospy.wait_for_service("/simulator/reset", timeout=init_timeout)
        rospy.wait_for_service("/simulator/start", timeout=init_timeout)
        # The existing evaluator performs an initial reset even if no init has arrived.
        _reset_simulator(reset, gate)
        for index in range(cfg.eval_episodes):
            generation_uri = _wait_init(gate, init_timeout)
            path = bags / f"episode_{index + 1:06d}.bag"
            recorder = _Recorder(path, topics, ready, shutdown)
            result = _episode(config, policy, preprocessor, postprocessor, recorder, tail,
                              generation_uri, observation_topics, max_observation_age,
                              observation_ready_timeout, environment_init_timeout)
            result.update({"episode": index + 1, "seed": cfg.seed,
                           "checkpoint": str(model_dir or cfg.pretrained_path),
                           "bag": str(path), "simulator_reset_service_uri": generation_uri,
                           "compatibility": "not_verified"})
            try:
                result["bag_inspection"] = _inspect_bag(path, required, expected_types)
            except Exception as exc:
                result["bag_error"] = f"{type(exc).__name__}: {exc}"
            result["candidate_for_review"] = bool(
                result["stop_reason"] == "success" and result.get("rosbag_exit_code") == 0
                and not result.get("recording_error") and not result.get("cleanup_error")
                and not result.get("bag_error")
                and not result.get("bag_inspection", {}).get("missing_required_topics")
                and not result.get("bag_inspection", {}).get("type_mismatches")
            )
            _write_json(metadata / f"episode_{index + 1:06d}.json", result)
            results.append(result)
            rospy.loginfo("Flywheel episode %s: %s; candidate=%s", index + 1,
                          result["stop_reason"], result["candidate_for_review"])
            if (result["stop_reason"] in {"error", "interrupted"}
                    or result.get("recording_error") or result.get("cleanup_error")
                    or result.get("bag_error")
                    or result.get("bag_inspection", {}).get("missing_required_topics")
                    or result.get("bag_inspection", {}).get("type_mismatches")
                    or result.get("rosbag_exit_code") != 0):
                collection_error = "Episode failed health or recording checks; remaining episodes not started"
                break
            if index + 1 < cfg.eval_episodes:
                _reset_simulator(reset, gate)
            else:
                # Match ordinary auto-test: end the completed final simulator
                # round. deploy.py may automatically start the next round.
                final_reset["attempted"] = True
                if _service_uri("/simulator/reset") != generation_uri:
                    raise RuntimeError("Simulator generation changed before final reset")
                response = reset(TriggerRequest())
                if not response.success:
                    raise RuntimeError(f"Final simulator reset failed: {response.message}")
                final_reset["success"] = True
    except BaseException as exc:
        collection_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        service.shutdown()
        _write_json(run / "collection_summary.json", {
            "requested_episodes": cfg.eval_episodes, "recorded_episodes": len(results),
            "successes": sum(r["stop_reason"] == "success" for r in results),
            "candidates_for_review": sum(bool(r["candidate_for_review"]) for r in results),
            "collection_error": collection_error,
            "final_reset": final_reset,
            "note": "Candidates require conversion, action alignment and release-label verification.",
        })
        rospy.loginfo("Flywheel output: %s", run)
    if collection_error:
        raise RuntimeError(collection_error)

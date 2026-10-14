#!/usr/bin/env bash
# Current three-robot scenario; own and stop only this launch's process groups.
set -eo pipefail
MARCO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROS_ADAPTER="$MARCO_ROOT/src/simulations/ros2"
ROS_SCENARIO="$ROS_ADAPTER/three_robot_t"
export MARCO_PYTHON="${MARCO_PYTHON:-$(command -v python3)}"
if [[ -n "${MARCO_ROS_SETUP:-}" ]]; then
  source "$MARCO_ROS_SETUP"
fi
command -v ros2 >/dev/null || { echo 'Source ROS 2 Humble or set MARCO_ROS_SETUP.'; exit 1; }
command -v gzserver >/dev/null || { echo 'Gazebo Classic is required.'; exit 1; }
: "${MARCO_ROS_MODEL_PACKAGE:?Set this to the external multi_robot model package directory}"
test -f "$MARCO_ROS_MODEL_PACKAGE/models/turtlebot3_waffle/model.sdf" || {
  echo 'Missing current scenario robot model.sdf.'; exit 1;
}
export PYTHONPATH="$MARCO_ROOT/src:$ROS_ADAPTER:$ROS_SCENARIO${PYTHONPATH:+:$PYTHONPATH}"
export ROS_DOMAIN_ID=73 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$ROS_ADAPTER/cyclonedds.xml"
export GAZEBO_MASTER_URI=http://127.0.0.1:11473 GAZEBO_MODEL_DATABASE_URI=''
export GAZEBO_MODEL_PATH="/usr/share/gazebo-11/models:$MARCO_ROS_MODEL_PACKAGE/models${GAZEBO_MODEL_PATH:+:$GAZEBO_MODEL_PATH}"
export DEMO_START_MODE="${DEMO_START_MODE:-approach}"
if ss -ltn 'sport = :11473' | tail -n +2 | grep -q .; then
  echo 'Gazebo port 11473 is occupied. Stop the previous demo first.'; exit 1
fi
exec 9>/tmp/marco_ros_three_robot.lock
flock -n 9 || { echo 'Three-robot run is already active.'; exit 1; }
mkdir -p "$MARCO_ROOT/results/ros2"
export DEMO_REPORT_DIR="$(mktemp -d "$MARCO_ROOT/results/ros2/run_XXXXXXXX")"
"$MARCO_PYTHON" "$ROS_ADAPTER/prepare_map.py" \
  "$ROS_SCENARIO/measured_images/scanned_map.yaml" "$DEMO_REPORT_DIR/scan_audit.json"
echo "Run outputs: $DEMO_REPORT_DIR"
groups=()
cleanup() {
  trap - EXIT INT TERM
  for group in "${groups[@]}"; do kill -INT -- "-$group" 2>/dev/null || true; done
  sleep 3
  for group in "${groups[@]}"; do kill -TERM -- "-$group" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
setsid ros2 launch "$ROS_SCENARIO/rviz_capture.launch.py" >"$DEMO_REPORT_DIR/ros_launch.log" 2>&1 &
groups+=("$!")
sleep 8
setsid timeout --signal=INT --kill-after=10s 360s /usr/bin/python3 -u \
  "$ROS_SCENARIO/execute.py" > >(tee "$DEMO_REPORT_DIR/coordinator.log") 2>&1 &
groups+=("$!")
result=0
wait "${groups[1]}" || result=$?
if [[ "$result" -ne 0 ]]; then exit "$result"; fi
echo "Execution complete. Audit: $DEMO_REPORT_DIR/execution_audit.json"

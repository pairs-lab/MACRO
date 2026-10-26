#!/usr/bin/env bash
# Three-robot T-junction with TurtleBot 4 (Ignition Gazebo, no charging dock); own and stop only this launch's process groups.
set -eo pipefail
MARCO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROS_ADAPTER="$MARCO_ROOT/src/simulations/ros2"
ROS_SCENARIO="$ROS_ADAPTER/three_robot_t"
# A TurtleBot 4 simulator environment: ROS 2 Humble plus turtlebot4_ignition_bringup, irobot_create_*, nav2_bringup.
TB4_WORKSPACE="${MARCO_TB4_WORKSPACE:-$HOME/workspaces/turtlebot}"
[[ -f "$TB4_WORKSPACE/environment.sh" ]] || { echo "Set MARCO_TB4_WORKSPACE to a TurtleBot 4 simulator workspace (environment.sh)."; exit 1; }
source "$TB4_WORKSPACE/environment.sh"
set +u
for pkg in turtlebot4_ignition_bringup irobot_create_control nav2_bringup nav2_simple_commander; do
  ros2 pkg prefix "$pkg" >/dev/null 2>&1 || { echo "Missing ROS package $pkg."; exit 1; }
done
if [[ -z "${MARCO_PYTHON:-}" ]]; then
  MARCO_PYTHON="$HOME/miniconda3/envs/ai2thor/bin/python"; [[ -x "$MARCO_PYTHON" ]] || MARCO_PYTHON="$(command -v python3)"
fi
export MARCO_PYTHON
export MARCO_ROBOT=tb4 MARCO_BASE_FRAME=base_link
export MARCO_TB4_CONTROL_YAML="$(ros2 pkg prefix irobot_create_control)/share/irobot_create_control/config/control.yaml"
export PYTHONPATH="$MARCO_ROOT/src:$ROS_ADAPTER:$ROS_SCENARIO${PYTHONPATH:+:$PYTHONPATH}"
export ROS_DOMAIN_ID=73 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$ROS_ADAPTER/tb4/cyclonedds_tb4.xml"
export DEMO_START_MODE="${DEMO_START_MODE:-approach}"
export MARCO_TB4_HEADLESS="${MARCO_TB4_HEADLESS:-true}"
if pgrep -f "[i]gn gazebo.*three_robot_t.sdf" >/dev/null; then
  echo 'A TurtleBot 4 T-junction simulation is already running. Stop it first.'; exit 1
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
  sleep 4
  for group in "${groups[@]}"; do kill -TERM -- "-$group" 2>/dev/null || true; done
  sleep 2
  for group in "${groups[@]}"; do kill -KILL -- "-$group" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
setsid ros2 launch "$ROS_ADAPTER/tb4/launch/tb4_scenario.launch.py" >"$DEMO_REPORT_DIR/ros_launch.log" 2>&1 &
groups+=("$!")
sleep "${MARCO_TB4_STARTUP_S:-45}"
setsid timeout --signal=INT --kill-after=10s 420s /usr/bin/python3 -u \
  "$ROS_SCENARIO/execute.py" > >(tee "$DEMO_REPORT_DIR/coordinator.log") 2>&1 &
groups+=("$!")
result=0
wait "${groups[1]}" || result=$?
if [[ "$result" -ne 0 ]]; then exit "$result"; fi
echo "Execution complete. Audit: $DEMO_REPORT_DIR/execution_audit.json"

# ROS 2 / Nav2 / Gazebo adapter

The current scenario is `three_robot_t/`: two robots exchange sides of a narrow T-junction while a third robot provides composed clearance. Nav2 and AMCL execute the navigation, with all three robots, inflation and bounds displayed in one RViz window.

## Prerequisites

- ROS 2 Humble, Nav2, RViz2, Gazebo Classic 11 and Cyclone DDS.
- The external `multi_robot` package's current robot SDF, URDF and meshes. This scenario currently uses its `turtlebot3_waffle` Gazebo model, not a physical TurtleBot 4 driver.
- A Python environment with `requirements-ros2.txt` and both native modules built.

The robot assets can be supplied from the existing `ros2-multi-robot-automap` workspace. They are not duplicated or downloaded by this launcher.

## Launch

From the repository root, configure paths for your installation:

```bash
export MARCO_ROS_SETUP=/path/to/ros/environment.sh
export MARCO_ROS_MODEL_PACKAGE=/path/to/ros2-multi-robot-automap/src/multi_robot
export MARCO_PYTHON=/path/to/python/environment/bin/python
bash run_ros2.sh
```

If ROS is already sourced, omit `MARCO_ROS_SETUP`. ROS nodes run with system Python; the persistent planning worker uses `MARCO_PYTHON`. ROS domain 73 and Gazebo port 11473 are isolated for this scenario. The launcher stops only its own process groups.

Outputs are written to ignored `results/ros2/run_*/`, including actual launch parameters, selected intervention contracts and execution logs. The saved map under `three_robot_t/measured_images/` is the AMCL input; `prepare_map.py` converts that same map to planning occupancy at startup. This launcher does not run a new SLAM scan.

## TurtleBot 4 (Ignition Gazebo, no charging dock)

`run_ros2_tb4.sh` runs the same T-junction scenario, Nav2/AMCL setup, MARCO planner and executor with three TurtleBot 4
(Create 3) robots in Ignition Gazebo. No dock is spawned.

Prerequisites: ROS 2 Humble with a TurtleBot 4 simulator environment (`turtlebot4_ignition_bringup`, `irobot_create_*`,
`nav2_bringup`, `nav2_simple_commander`), reachable through the `environment.sh` of a TurtleBot 4 workspace.

```bash
export MARCO_TB4_WORKSPACE=/path/to/turtlebot4/workspace   # contains environment.sh
export MARCO_PYTHON=/path/to/python/environment/bin/python  # numpy + native modules (planning worker)
bash run_ros2_tb4.sh
```

`MARCO_TB4_HEADLESS=false` additionally opens the Ignition window; `MARCO_TB4_MODEL=standard` selects the standard model.

What differs from the TurtleBot 3 run, and why:

- `tb4/worlds/three_robot_t.sdf`: the same walls as `three_robot_t.world`, with the Ignition Sensors system declared once.
- `tb4/make_robot_description.py`: the official TurtleBot 4 description without its per-robot Sensors/Contact plugins
  (several robots with one each crash Ignition) and without the cliff/IR/camera render sensors. RPLIDAR, wheel odometry,
  IMU, bumper contact and the diff-drive controller remain.
- `tb4/launch/tb4_robot.launch.py`: robot_state_publisher, entity, the two ros2_control spawners and two frame aliases per
  robot. The official bringup starts about 65 processes per robot (one ros_gz bridge per topic); three robots saturated a
  20-thread laptop (load average above 200). The scan and clock are bridged once (`tb4_world.launch.py`) and
  `tb4/cmd_vel_relay.py` forwards `/tbN/cmd_vel` to the diff-drive controller.
- `tb4/cyclonedds_tb4.xml`: more CycloneDDS participants than the default 100.
- Base frame `base_link` (`MARCO_BASE_FRAME`), round 0.185 m footprint, wheel radius/separation/acceleration read from
  `irobot_create_control/config/control.yaml` for the timing prediction (`MARCO_ROBOT=tb4`).
- `tb4/launch/tb4_scenario.launch.py` adapts `nav2_tb1.yaml` and reuses `marco_controllers.py` (the shared controller profile).

Execution (both robots): an action starts when its route is disjoint from the remaining routes of earlier unfinished
actions of other robots, and the prediction (`schedule_staging`) uses the same rule; R3 enters the pocket in one smooth
path (`DEMO_R3_ENTRY=two_stage` restores the stop at the mouth). Bounded recoveries (pose corrections, progress
watchdog, DWB retries) are recorded in `prediction_contract_deviations`.

## Module boundaries

- `map_adapter.py`: occupancy-to-grid input and static geometry caching.
- `marco_geometry.py`: metric adapter using the shared core swept-distance primitive.
- `planner_worker.py`: persistent planning-process transport.
- `three_robot_t/planner.py`: core candidate search plus ROS execution-contract adaptation before selection.
- `three_robot_t/execution_prediction.py`: controller-aware navigation timing prediction.
- `three_robot_t/execute.py`: Nav2 actions, observed release gates and safety checks.
- `fleet_bridge.py`, `fleet_obstacles.py`: visualization and peer-footprint costmap adapters.

The ROS timing objective covers this navigation scenario, not general task-service DAGs. This is a Gazebo integration, not evidence of physical-robot accuracy or a Nav2/Gazebo physics clone.

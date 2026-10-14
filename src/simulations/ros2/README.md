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

## Module boundaries

- `map_adapter.py`: occupancy-to-grid input and static geometry caching.
- `marco_geometry.py`: metric adapter using the shared core swept-distance primitive.
- `planner_worker.py`: persistent planning-process transport.
- `three_robot_t/planner.py`: core candidate search plus ROS execution-contract adaptation before selection.
- `three_robot_t/execution_prediction.py`: controller-aware navigation timing prediction.
- `three_robot_t/execute.py`: Nav2 actions, observed release gates and safety checks.
- `fleet_bridge.py`, `fleet_obstacles.py`: visualization and peer-footprint costmap adapters.

The ROS timing objective covers this navigation scenario, not general task-service DAGs. This is a Gazebo integration, not evidence of physical-robot accuracy or a Nav2/Gazebo physics clone.

"""OpenCV visualization and input handling, separate from grid execution."""
import math
import cv2
import numpy as np


class GridRenderer:
    @staticmethod
    def _robot_display_cell(robot):
        """Interpolate edge progress for display only; never change grid state.

        The executor remains authoritative for discrete cell commits, swept
        collision checks and release gates. This is not continuous physics.
        """
        edge = getattr(robot, 'movement_edge', None)
        if edge is None or tuple(edge[0]) != tuple(robot.cell):
            return robot.cell
        length = math.dist(*edge)
        if length <= 1e-12:
            return robot.cell
        fraction = min(1., max(0., getattr(robot, 'movement_credit', 0.) / length))
        return tuple(a + fraction*(b-a) for a, b in zip(*edge))

    def _pixel(self, cell):
        min_x, _, _, max_z = self.bounds
        return (
            int(round(self.offset_x + (cell[0] - min_x + 0.5) * self.scale)),
            int(round(self.offset_y + (max_z - cell[1] + 0.5) * self.scale)),
        )


    def _cell_from_pixel(self, pixel_x, pixel_y):
        """Convert a window pixel to the nearest JSON grid coordinate."""
        min_x, _, _, max_z = self.bounds
        cell_x = math.floor((pixel_x - self.offset_x) / self.scale) + min_x
        cell_z = max_z - math.floor((pixel_y - self.offset_y) / self.scale)
        return int(cell_x), int(cell_z)


    @staticmethod
    def _draw_dashed_path(frame, pixels, color, thickness=3, dash=8, gap=5):
        """Draw a polyline as evenly spaced anti-aliased dashes."""
        pattern = dash + gap
        travelled = 0.0
        for start, end in zip(pixels, pixels[1:]):
            start = np.asarray(start, dtype=float)
            end = np.asarray(end, dtype=float)
            vector = end - start
            length = float(np.linalg.norm(vector))
            if length == 0.0:
                continue
            direction = vector / length
            offset = 0.0
            while offset < length:
                phase = (travelled + offset) % pattern
                step = min(length - offset, pattern - phase)
                if phase < dash:
                    draw_length = min(step, dash - phase)
                    point_a = tuple(np.rint(start + direction * offset).astype(int))
                    point_b = tuple(
                        np.rint(start + direction * (offset + draw_length)).astype(int)
                    )
                    cv2.line(
                        frame,
                        point_a,
                        point_b,
                        color,
                        thickness,
                        cv2.LINE_AA,
                    )
                offset += step
            travelled += length


    def _on_mouse(self, event, pixel_x, pixel_y, _flags, _parameter):
        if pixel_x >= self.dag_left:
            self.hover_cell = None
            return
        cell = self._cell_from_pixel(pixel_x, pixel_y)
        self.hover_cell = cell
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        self.selected_cell = cell
        world_x = cell[0] * self.world.grid_size
        world_z = cell[1] * self.world.grid_size
        if cell in self.world.navigation_cells:
            state = "NAVIGABLE"
        elif cell in self.world.free_cells:
            state = "ROBOT_CLEARANCE"
        else:
            state = "BLOCKED"
        print(
            f"Selected map coordinate: cell=[{cell[0]}, {cell[1]}], "
            f"world_m=[{world_x:.2f}, {world_z:.2f}], state={state}"
        )


    def _draw_map(self):
        frame = np.full((self.height, self.width, 3), 32, dtype=np.uint8)
        for cell in self.world.free_cells:
            centre = self._pixel(cell)
            half = self.scale / 2
            cv2.rectangle(
                frame,
                (int(centre[0] - half), int(centre[1] - half)),
                (int(centre[0] + half), int(centre[1] + half)),
                (248, 248, 248),
                -1,
            )
        if self.scale >= 5:
            for cell in self.world.free_cells:
                centre = self._pixel(cell)
                half = self.scale / 2
                cv2.rectangle(
                    frame,
                    (int(centre[0] - half), int(centre[1] - half)),
                    (int(centre[0] + half), int(centre[1] + half)),
                    (220, 220, 220),
                    1,
                )
        return frame


    def _draw(self):
        frame = self.background.copy()
        visible_objects = sorted(
            self.objects.values(),
            key=lambda value: bool(value.get("contained_by")),
        )
        contained_counts = {}
        for item in visible_objects:
            if not item.get("visible", True):
                continue
            color = tuple(item.get("color_bgr", (80, 180, 230)))
            if item.get("shape", "point") == "segment":
                start = self._pixel(item["start_cell"])
                end = self._pixel(item["end_cell"])
                thickness = max(
                    3,
                    int((2 * item.get("thickness_cells", 0) + 1) * self.scale),
                )
                cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)
                centre = self._pixel(item["cell"])
                radius = thickness // 2
            else:
                display_cell = item["cell"]
                container = item.get("contained_by")
                if container:
                    index = contained_counts.get(container, 0)
                    contained_counts[container] = index + 1
                    display_cell = (
                        display_cell[0] + (-1 if index % 2 == 0 else 1),
                        display_cell[1] + index // 2,
                    )
                centre = self._pixel(display_cell)
                radius = max(5, int(item.get("radius_cells", 1) * self.scale))
                cv2.circle(frame, centre, radius, color, -1, cv2.LINE_AA)
            state_suffix = (
                " (OPEN)"
                if item.get("is_open")
                else " (ON)" if item.get("is_on") else ""
            )
            cv2.putText(
                frame,
                item["name"] + state_suffix,
                (centre[0] + radius + 3, centre[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                item["name"] + state_suffix,
                (centre[0] + radius + 3, centre[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        radius = max(
            7,
            int(self.world.robot_radius / self.world.grid_size * self.scale),
        )
        for robot in self.robots:
            display_cell = self._robot_display_cell(robot)
            if (
                self.show_paths
                and (self._is_navigating(robot) or robot.phase == "preview")
                and robot.path_index < len(robot.path) - 1
            ):
                pixels = np.array(
                    [self._pixel(display_cell)] +
                    [self._pixel(cell) for cell in robot.path[robot.path_index + 1:]],
                    dtype=np.int32,
                )
                if robot.phase == "preview":
                    self._draw_dashed_path(frame, pixels, robot.color)
                else:
                    cv2.polylines(
                        frame,
                        [pixels],
                        False,
                        robot.color,
                        2,
                        cv2.LINE_AA,
                    )
            centre = self._pixel(display_cell)
            cv2.circle(frame, centre, radius, robot.color, -1, cv2.LINE_AA)
            cv2.circle(frame, centre, radius, (10, 10, 10), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                robot.robot_id,
                (centre[0] + radius + 4, centre[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (10, 10, 10),
                2,
                cv2.LINE_AA,
            )
            if robot.phase == "acting":
                cv2.putText(
                    frame,
                    f"{robot.current_task['action']} {robot.action_remaining:.1f}s",
                    (centre[0] + radius + 4, centre[1] + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            elif robot.phase == "preview":
                cv2.putText(
                    frame,
                    f"preferred path {robot.preview_remaining:.1f}s",
                    (centre[0] + radius + 4, centre[1] + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        self._draw_dag(frame)
        inspected_cell = self.selected_cell or self.hover_cell
        if inspected_cell is not None:
            centre = self._pixel(inspected_cell)
            if inspected_cell in self.world.navigation_cells:
                state = "NAVIGABLE"
            elif inspected_cell in self.world.free_cells:
                state = "ROBOT_CLEARANCE"
            else:
                state = "BLOCKED"
            world_x = inspected_cell[0] * self.world.grid_size
            world_z = inspected_cell[1] * self.world.grid_size
            cv2.drawMarker(
                frame,
                centre,
                (0, 0, 255),
                cv2.MARKER_CROSS,
                max(12, int(self.scale * 2)),
                2,
                cv2.LINE_AA,
            )
            cv2.rectangle(
                frame,
                (14, self.height - 42),
                (self.map_width - 14, self.height - 10),
                (15, 15, 15),
                -1,
            )
            cv2.putText(
                frame,
                (
                    f"cell=[{inspected_cell[0]}, {inspected_cell[1]}]  "
                    f"world=[{world_x:.2f}, {world_z:.2f}] m  {state}"
                ),
                (24, self.height - 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        status = "PAUSED" if self.paused else "RUNNING"
        cv2.putText(
            frame,
            f"{status} | Click: pin coordinate | Space: pause | R: reset | "
            "P: paths | Q/Esc: quit",
            (20, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        return frame


    def _task_depths(self):
        tasks = {task["id"]: task for task in self.tasks}
        depths = {}

        def depth(task_id):
            if task_id in depths:
                return depths[task_id]
            dependencies = tasks[task_id]["node"]
            depths[task_id] = (
                max((depth(predecessor) for predecessor in dependencies), default=-1)
                + 1
            )
            return depths[task_id]

        for task_id in tasks:
            depth(task_id)
        return depths


    def _draw_dag(self, frame):
        """Draw the live dependency graph beside the occupancy map."""
        cv2.rectangle(
            frame,
            (self.dag_left, 0),
            (self.width - 1, self.height - 1),
            (24, 24, 24),
            -1,
        )
        cv2.line(
            frame,
            (self.dag_left, 0),
            (self.dag_left, self.height),
            (90, 90, 90),
            2,
        )
        cv2.putText(
            frame,
            "LIVE TASK DAG",
            (self.dag_left + 18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if not self.tasks:
            return

        depths = self._task_depths()
        robot_ids = [robot.robot_id for robot in self.robots]
        lane_width = (self.width - self.dag_left - 24) / max(1, len(robot_ids))
        max_depth = max(depths.values(), default=0)
        level_height = min(88.0, (self.height - 100) / max(1, max_depth + 1))
        centres = {}
        for task in self.tasks:
            lane = robot_ids.index(task["robot"])
            centres[task["id"]] = (
                int(self.dag_left + 12 + (lane + 0.5) * lane_width),
                int(70 + (depths[task["id"]] + 0.5) * level_height),
            )

        for task in self.tasks:
            end = centres[task["id"]]
            for predecessor in task["node"]:
                start = centres[predecessor]
                cv2.arrowedLine(
                    frame,
                    (start[0], start[1] + 22),
                    (end[0], end[1] - 22),
                    (150, 150, 150),
                    1,
                    cv2.LINE_AA,
                    tipLength=0.12,
                )

        status_colors = {
            "pending": (75, 75, 75),
            "planned": (160, 80, 180),
            "running": (210, 120, 20),
            "acting": (20, 150, 240),
            "completed": (55, 170, 70),
        }
        node_half_width = max(42, int(lane_width / 2 - 7))
        for task in self.tasks:
            centre = centres[task["id"]]
            status = self.task_status.get(task["id"], "pending")
            color = status_colors.get(status, (75, 75, 75))
            cv2.rectangle(
                frame,
                (centre[0] - node_half_width, centre[1] - 21),
                (centre[0] + node_half_width, centre[1] + 21),
                color,
                -1,
            )
            cv2.rectangle(
                frame,
                (centre[0] - node_half_width, centre[1] - 21),
                (centre[0] + node_half_width, centre[1] + 21),
                (225, 225, 225),
                1,
            )
            cv2.putText(
                frame,
                f"T{task['id']} {task['robot']}",
                (centre[0] - node_half_width + 5, centre[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{task['action']} {task['object']}",
                (centre[0] - node_half_width + 5, centre[1] + 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


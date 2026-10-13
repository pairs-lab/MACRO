class EndpointOracle:
    """Two approach poses: the deterministic endpoint depends on arrival."""
    def __init__(self):
        self.starts = {"R1": (0, 0), "R2": (10, 0)}
        self.goals = {"a": ((1, 0), (9, 0)), "b": ((2, 0),), "c": ((8, 0),)}

    def plan(self, robot, location, start_position=None):
        rid = "R" + robot[5:] if robot.startswith("robot") else robot
        start = self.starts[rid] if start_position is None else start_position
        goal = min(self.goals[location], key=lambda p: (abs(p[0]-start[0]),p))
        return abs(goal[0]-start[0]), goal

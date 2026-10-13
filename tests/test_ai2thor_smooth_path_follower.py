import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _load_follower_module():
    """Load the follower without requiring the optional AI2-THOR package."""
    source = (
        Path(__file__).resolve().parents[1]
        / "src/simulations/ai2thor/navigation/smooth_path_follower.py"
    )
    cameras = types.ModuleType("simulations.ai2thor.cameras")
    cameras.move_robot = lambda *args, **kwargs: None
    spec = importlib.util.spec_from_file_location(
        "_smooth_path_follower_under_test",
        source,
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"simulations.ai2thor.cameras": cameras}):
        spec.loader.exec_module(module)
    return module


def _load_navigation_module():
    """Load grid-navigation helpers with optional Unity modules stubbed."""
    source = (
        Path(__file__).resolve().parents[1]
        / "src/simulations/ai2thor/navigation/navigation.py"
    )
    stubs = {}
    cameras = types.ModuleType("simulations.ai2thor.cameras")
    cameras.aim_robot_camera = lambda *args, **kwargs: None
    cameras.move_robot = lambda *args, **kwargs: None
    stubs[cameras.__name__] = cameras
    controller = types.ModuleType("simulations.ai2thor.controller")
    controller.require_success = lambda event, _operation: event
    stubs[controller.__name__] = controller
    grid = types.ModuleType("simulations.ai2thor.navigation.grid_planner")
    grid.find_padded_astar_path = lambda *args, **kwargs: ([], {})
    grid.get_reachable_positions = lambda *args, **kwargs: []
    stubs[grid.__name__] = grid
    doors = types.ModuleType("simulations.ai2thor.navigation.door_rules")
    doors.future_open_rule = lambda _name: None
    stubs[doors.__name__] = doors
    smooth = types.ModuleType(
        "simulations.ai2thor.navigation.smooth_path_follower"
    )
    smooth.follow_smooth_path = lambda *args, **kwargs: None
    stubs[smooth.__name__] = smooth
    spec = importlib.util.spec_from_file_location(
        "simulations.ai2thor.navigation._navigation_under_test",
        source,
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class SmoothPathFollowerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.follower = _load_follower_module()

    @staticmethod
    def _event(*, carrying):
        inventory = [{"objectType": "Bowl"}] if carrying else []
        return SimpleNamespace(metadata={"inventoryObjects": inventory})

    def test_carried_payload_tracks_one_dense_astar_sample(self):
        lookahead, move_step = self.follower._tracking_limits(
            self._event(carrying=True),
            0.18,
        )
        self.assertEqual(lookahead, 0.05)
        self.assertEqual(move_step, 0.05)

    def test_unloaded_robot_keeps_smooth_tracking_parameters(self):
        lookahead, move_step = self.follower._tracking_limits(
            self._event(carrying=False),
            0.18,
        )
        self.assertEqual(lookahead, 0.18)
        self.assertEqual(move_step, 0.10)

    def test_payload_clearance_includes_observed_hand_extension(self):
        from simulations.ai2thor.config import carried_object_safety_margin
        metadata = {
            "agent": {"position": {"x": 0.0, "z": 0.0}},
            "inventoryObjects": [{"objectId": "Spoon|1", "objectType": "Spoon"}],
            "objects": [{"objectId": "Spoon|1", "axisAlignedBoundingBox": {
                "center": {"x": 0.0, "z": 0.5},
                "size": {"x": 0.04, "z": 0.2},
            }}],
        }
        self.assertGreater(carried_object_safety_margin(metadata), 0.48)
        metadata["inventoryObjects"][0]["objectType"] = "Apple"
        metadata["objects"][0]["axisAlignedBoundingBox"]["center"]["z"] = 0.15
        self.assertLess(carried_object_safety_margin(metadata), 0.20)
        metadata["inventoryObjects"] = []
        self.assertEqual(carried_object_safety_margin(metadata), 0.0)

    def test_translation_cannot_overshoot_next_astar_sample(self):
        magnitude = self.follower._bounded_move_magnitude(
            {"x": 0.0, "z": 0.0},
            {"x": 0.0, "z": 0.05},
            {"x": 0.5, "z": 0.5},
            0.10,
        )
        self.assertAlmostEqual(magnitude, 0.05)

    def test_selected_intervention_uses_safety_clearance_without_retrigger_buffer(self):
        events = [
            SimpleNamespace(metadata={
                "agent": {"position": {"x": 0.0, "z": 0.60}},
                "inventoryObjects": [],
            }),
            SimpleNamespace(metadata={
                "agent": {"position": {"x": 2.0, "z": 2.0}},
                "inventoryObjects": [],
            }),
            SimpleNamespace(metadata={
                "agent": {"position": {"x": 0.0, "z": 0.0}},
                "inventoryObjects": [{"objectType": "Spoon"}],
            }),
        ]
        controller = SimpleNamespace(last_event=SimpleNamespace(events=events))

        self.assertEqual(
            self.follower._agent_blocking_ahead(
                controller,
                2,
                events[2].metadata["agent"]["position"],
                {"x": 0.0, "z": 1.0},
            ),
            0,
        )
        self.assertIsNone(
            self.follower._agent_blocking_ahead(
                controller,
                2,
                events[2].metadata["agent"]["position"],
                {"x": 0.0, "z": 1.0},
                observation_exempt_agent_ids={0},
            )
        )


class InteractionGridCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.navigation = _load_navigation_module()

    def test_faucet_fallback_uses_real_visible_poses_and_caches_them(self):
        obj={'objectId':'Faucet|1','objectType':'Faucet',
             'position':{'x':1.,'y':1.,'z':1.}}
        metadata={'objects':[obj],'agent':{'position':{'x':0.,'y':0.,'z':0.}},'inventoryObjects':[]}
        calls=[]
        controller=SimpleNamespace(last_event=SimpleNamespace(metadata=metadata))
        def step(**kwargs):
            calls.append(kwargs)
            poses=[] if kwargs['horizons']==[30] else [{'x':1.4,'y':0.,'z':1.,'horizon':0}]
            controller.last_event=SimpleNamespace(metadata=dict(metadata,actionReturn=poses))
            return controller.last_event
        controller.step=step
        cache={}; timing={}
        first=self.navigation._interaction_goals(controller,'faucet',raw_pose_cache=cache,timing=timing)
        second=self.navigation._interaction_goals(controller,'faucet',raw_pose_cache=cache)
        self.assertEqual(first,second)
        self.assertEqual([c['horizons'] for c in calls],[[30],[0,60]])
        self.assertEqual(timing['unity_goal_queries'],2)
        self.assertTrue(first[0][0]['interaction_pose'])
        self.assertTrue(all('forceAction' not in c for c in calls))

    def test_faucet_fallback_never_accepts_invisible_or_out_of_range_poses(self):
        obj={'objectId':'Faucet|1','objectType':'Faucet','position':{'x':1.,'y':1.,'z':1.}}
        metadata={'objects':[obj],'agent':{'position':{'x':0.,'y':0.,'z':0.}},'inventoryObjects':[]}
        for fallback in ([],[{'x':99.,'y':0.,'z':99.}]):
            controller=SimpleNamespace(last_event=SimpleNamespace(metadata=metadata))
            def step(**kwargs):
                poses=[] if kwargs['horizons']==[30] else fallback
                controller.last_event=SimpleNamespace(metadata=dict(metadata,actionReturn=poses))
                return controller.last_event
            controller.step=step
            with self.assertRaisesRegex(RuntimeError,'No reachable pose'):
                self.navigation._interaction_goals(controller,'faucet')

    def test_scene_faucet_range_override_keeps_other_objects_at_default_range(self):
        interaction_distance_for = self.navigation.interaction_distance_for
        overrides = interaction_distance_for.__globals__['INTERACTION_DISTANCE_OVERRIDES']
        obj={'objectId':'Faucet|1','objectType':'Faucet','position':{'x':1.,'y':1.,'z':1.}}
        metadata={'objects':[obj],'agent':{'position':{'x':0.,'y':0.,'z':0.}},'inventoryObjects':[]}
        controller=SimpleNamespace(last_event=SimpleNamespace(metadata=metadata))
        def step(**kwargs):
            controller.last_event=SimpleNamespace(metadata=dict(metadata,actionReturn=[{'x':1.8,'y':0.,'z':1.}]))
            return controller.last_event
        controller.step=step
        with patch.dict(overrides,{},clear=True):
            with self.assertRaisesRegex(RuntimeError,'nearest=0.800m'):
                self.navigation._interaction_goals(controller,'faucet')
        with patch.dict(overrides,{'faucet':1.},clear=True):
            self.assertTrue(self.navigation._interaction_goals(controller,'faucet'))
            self.assertEqual(interaction_distance_for(obj),1.)
            self.assertEqual(interaction_distance_for({'objectType':'Bowl'}),.7)

    def test_raw_interaction_region_is_shared_across_identical_agents(self):
        scene_object = {
            "objectId": "Bowl|1",
            "objectType": "Bowl",
            "position": {"x": 1.0, "y": 0.0, "z": 1.0},
            "axisAlignedBoundingBox": {
                "center": {"x": 1.0, "y": 0.1, "z": 1.0},
                "size": {"x": 0.2, "y": 0.2, "z": 0.2},
            },
            "isOpen": False,
        }
        metadata = {
            "lastActionSuccess": True,
            "objects": [scene_object],
            "agent": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
            "inventoryObjects": [],
        }

        class Controller:
            def __init__(self):
                self.last_event = SimpleNamespace(metadata=metadata)
                self.queries = 0

            def step(self, **_kwargs):
                self.queries += 1
                result = dict(metadata)
                result["actionReturn"] = [
                    {"x": 1.4, "y": 0.0, "z": 1.0}
                ]
                self.last_event = SimpleNamespace(metadata=result)
                return self.last_event

        controller = Controller()
        raw_cache = {}
        first = self.navigation._interaction_goals(
            controller,
            "Bowl",
            agent_id=0,
            raw_pose_cache=raw_cache,
        )
        second = self.navigation._interaction_goals(
            controller,
            "Bowl",
            agent_id=1,
            raw_pose_cache=raw_cache,
        )

        self.assertEqual(controller.queries, 1)
        self.assertEqual(first, second)
        self.assertTrue(first[0][0]["interaction_pose"])

        # A user hint outside the reachable region must never become a fake
        # interactable pose or an unchecked terminal straight-line move.
        with patch.dict(self.navigation.INTERACTION_POSITION_OVERRIDES,
                        {"bowl": {"x": 99.0, "z": 99.0}}):
            overridden = self.navigation._interaction_goals(
                controller, "Bowl", raw_pose_cache=raw_cache,
            )
        self.assertEqual(overridden, first)
        with patch.dict(self.navigation.INTERACTION_POSITION_OVERRIDES,
                        {'bowl': {'x':99.,'z':99.,'required':True,'tolerance':.03}}):
            with self.assertRaisesRegex(RuntimeError,'required interaction point'):
                self.navigation._interaction_goals(controller,'Bowl',raw_pose_cache=raw_cache)
        with patch.dict(self.navigation.INTERACTION_POSITION_OVERRIDES,
                        {'bowl': {'x':1.41,'z':1.,'required':True,'tolerance':.03}}):
            required=self.navigation._interaction_goals(controller,'Bowl',raw_pose_cache=raw_cache)
        self.assertEqual(required,first)

    def test_carried_payload_collision_maps_to_owning_agent(self):
        events = [
            SimpleNamespace(metadata={
                "inventoryObjects": [{"objectType": "ButterKnife"}],
            }),
            SimpleNamespace(metadata={"inventoryObjects": []}),
            SimpleNamespace(metadata={
                "inventoryObjects": [{"objectType": "Spoon"}],
            }),
        ]
        controller = SimpleNamespace(
            last_event=SimpleNamespace(events=events)
        )
        blocker = self.navigation._blocking_agent_from_error(
            "Spoon_ab4315e1 is blocking the Agent from moving by "
            "(0.01, 0.00, 0.05) with ButterKnife_b89ea1af",
            requesting_agent_id=0,
            controller=controller,
        )

        self.assertEqual(blocker, 2)


if __name__ == "__main__":
    unittest.main()

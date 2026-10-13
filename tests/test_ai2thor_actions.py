import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _load_actions_module():
    source = (
        Path(__file__).resolve().parents[1]
        / "src/simulations/ai2thor/actions.py"
    )
    cameras = types.ModuleType("simulations.ai2thor.cameras")
    cameras.move_robot = lambda *args, **kwargs: None
    controller = types.ModuleType("simulations.ai2thor.controller")

    def require_success(event, operation):
        if not event.metadata["lastActionSuccess"]:
            raise RuntimeError(operation)
        return event

    controller.require_success = require_success
    doors = types.ModuleType(
        "simulations.ai2thor.navigation.door_rules"
    )
    doors.future_open_rule = lambda _name: None
    spec = importlib.util.spec_from_file_location(
        "simulations.ai2thor._actions_under_test",
        source,
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            cameras.__name__: cameras,
            controller.__name__: controller,
            doors.__name__: doors,
        },
    ):
        spec.loader.exec_module(module)
    return module


class CompoundPayloadRestorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actions_module = _load_actions_module()

    def test_pick_retracts_in_collision_checked_small_steps_and_refreshes_view(self):
        obj={'objectId':'Bread|1','objectType':'Bread','receptacleObjectIds':[]}
        for blocked_at in (None,3):
            calls=[]; frames=[]
            pickup=SimpleNamespace(metadata={'lastActionSuccess':True})
            def step(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(metadata={'lastActionSuccess':blocked_at is None or len(calls)<blocked_at,
                                                 'errorMessage':'collision'})
            actions=self.actions_module.Actions(SimpleNamespace(step=step),None,on_step=frames.append)
            actions._target_object=lambda *a,**kw:obj
            actions._require_interaction_pose=lambda *a,**kw:obj
            actions._step=lambda *a,**kw:pickup
            actions.pick('robot1','bread')
            self.assertEqual(len(calls),6 if blocked_at else 8)
            self.assertTrue(all(c['action']=='MoveHandBack' and 0<c['moveMagnitude']<=.05 for c in calls))
            if blocked_at:
                self.assertEqual([c['moveMagnitude'] for c in calls[2:]],
                                 [.05,.025,.0125,.00625])
            else:
                self.assertAlmostEqual(sum(c['moveMagnitude'] for c in calls),.4)
            self.assertTrue(all(c['renderImage'] and 'forceAction' not in c for c in calls))
            self.assertTrue(frames[-1].metadata['lastActionSuccess'])
            self.assertEqual(actions.held_object_ids[0],'Bread|1')

    def test_pick_retraction_can_continue_with_smaller_collision_checked_steps(self):
        obj={'objectId':'Tomato|1','objectType':'Tomato','receptacleObjectIds':[]}
        calls=[]; frames=[]
        def step(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(metadata={'lastActionSuccess':kwargs['moveMagnitude']<=.0125,
                                             'errorMessage':'collision'})
        actions=self.actions_module.Actions(SimpleNamespace(step=step),None,on_step=frames.append)
        actions._target_object=lambda *a,**kw:obj
        actions._require_interaction_pose=lambda *a,**kw:obj
        actions._step=lambda *a,**kw:SimpleNamespace(metadata={'lastActionSuccess':True})
        result=actions.pick('robot1','tomato')
        self.assertTrue(result.metadata['lastActionSuccess'])
        self.assertEqual([c['moveMagnitude'] for c in calls[:3]],[.05,.025,.0125])
        self.assertEqual(len(frames),32)
        self.assertTrue(all('forceAction' not in c for c in calls))

    def test_pick_filled_receptacle_does_not_shift_its_contents(self):
        obj={'objectId':'Bowl|1','objectType':'Bowl','receptacleObjectIds':['Apple|1']}
        calls=[]
        actions=self.actions_module.Actions(SimpleNamespace(step=lambda **kw:calls.append(kw)),None)
        actions._target_object=lambda *a,**kw:obj
        actions._require_interaction_pose=lambda *a,**kw:obj
        actions._step=lambda *a,**kw:SimpleNamespace(metadata={'lastActionSuccess':True})
        actions.pick('robot1','bowl')
        self.assertEqual(calls,[])

    def test_payload_stows_to_body_clearance_and_remains_compact_for_delivery(self):
        obj = {'objectId': 'Bread|1', 'objectType': 'Bread',
               'position': {'x': 0., 'z': .85},
               'axisAlignedBoundingBox': {'size': {'x': .2, 'z': .2}}}
        event = SimpleNamespace(metadata={
            'lastActionSuccess': True, 'objects': [obj],
            'agent': {'position': {'x': 0., 'z': 0.}, 'rotation': {'y': 0.}}})
        calls = []
        def step(**kwargs):
            calls.append(kwargs)
            obj['position']['z'] -= kwargs['moveMagnitude']
            return event
        actions = self.actions_module.Actions(SimpleNamespace(step=step, last_event=event), None)
        actions._target_object = lambda *a, **kw: obj
        actions._require_interaction_pose = lambda *a, **kw: obj
        actions._step = lambda *a, **kw: event
        actions.pick('robot1', 'bread')
        target = self.actions_module.AGENT_RADIUS + .1 + .03
        self.assertAlmostEqual(obj['position']['z'], target)
        count = len(calls)
        actions.stow_for_navigation('robot1')
        self.assertEqual(len(calls), count)
        # An externally extended payload is corrected before delivery.
        obj['position']['z'] += .2
        actions.stow_for_navigation('robot1')
        self.assertAlmostEqual(obj['position']['z'], target)
        self.assertTrue(all('forceAction' not in call for call in calls))

    def test_metadata_fallback_does_not_retract_again_on_delivery(self):
        event = SimpleNamespace(metadata={'lastActionSuccess': True})
        calls = []
        def step(**kwargs):
            calls.append(kwargs)
            return event
        actions = self.actions_module.Actions(SimpleNamespace(step=step, last_event=event), None)
        actions.held_object_ids[0] = 'Tomato|1'
        actions._retract_hand('robot1', 0, event)
        actions.stow_for_navigation('robot1')
        self.assertEqual(len(calls), 8)

    def test_microwave_retry_is_semantic_and_never_places_bread_on_aabb_top(self):
        for success in (True,False):
            microwave={'objectId':'Microwave|1','objectType':'Microwave','receptacle':True,
                       'receptacleObjectIds':['Bread|1'] if success else []}
            bread={'objectId':'Bread|1','objectType':'Bread','parentReceptacles':[]}
            calls=[]
            def step(**kwargs):
                calls.append(kwargs)
                ok=success and len(calls)==2
                return SimpleNamespace(metadata={'lastActionSuccess':ok,'errorMessage':'slot blocked',
                                                 'objects':[microwave,bread]})
            controller=SimpleNamespace(step=step)
            actions=self.actions_module.Actions(controller,None)
            actions.held_object_ids[0]='Bread|1'
            actions._objects=lambda agent_id:([microwave,bread],{})
            actions._find_object=lambda robot,name:microwave
            actions._require_interaction_pose=lambda robot,obj:obj
            if success:
                actions.place('robot1','microwave')
                self.assertNotIn(0,actions.held_object_ids)
            else:
                with self.assertRaisesRegex(RuntimeError,'stationary retry'):
                    actions.place('robot1','microwave')
                self.assertEqual(actions.held_object_ids[0],'Bread|1')
            self.assertEqual([c['action'] for c in calls],['PutObject','PutObject'])
            self.assertTrue(calls[1]['placeStationary'])
            self.assertTrue(all('forceAction' not in c for c in calls))

    def test_live_manipulation_uses_same_scene_range_but_still_requires_visibility(self):
        obj={'objectId':'Faucet|1','objectType':'Faucet','position':{'x':0.,'y':1.,'z':.8},'visible':True}
        agent={'position':{'x':0.,'y':0.,'z':0.},'rotation':{'y':0.}}
        actions=self.actions_module.Actions(SimpleNamespace(),None)
        actions._objects=lambda agent_id:([obj],agent)
        with patch.object(self.actions_module,'interaction_distance_for',return_value=.7):
            with self.assertRaisesRegex(RuntimeError,'must be within 0.70'):
                actions._require_interaction_pose('robot1',obj)
        with patch.object(self.actions_module,'interaction_distance_for',return_value=1.):
            self.assertEqual(actions._require_interaction_pose('robot1',obj),obj)
            obj['visible']=False
            with self.assertRaisesRegex(RuntimeError,'not visible'):
                actions._require_interaction_pose('robot1',obj)

    def test_kinematic_fallback_when_content_cannot_come_to_rest(self):
        bowl = {
            "objectId": "Bowl|1",
            "objectType": "Bowl",
            "axisAlignedBoundingBox": {
                "center": {"x": 1.0, "y": 0.5, "z": 1.0},
                "size": {"x": 0.2, "y": 0.1, "z": 0.2},
            },
        }
        tomato = {
            "objectId": "Tomato|1",
            "objectType": "Tomato",
            "axisAlignedBoundingBox": {
                "size": {"x": 0.08, "y": 0.08, "z": 0.08},
            },
        }

        class Controller:
            def __init__(self):
                self.calls = []
                self.last_event = self._event(True)

            @staticmethod
            def _event(success):
                return SimpleNamespace(metadata={
                    "lastActionSuccess": success,
                    "errorMessage": (
                        "" if success else "object couldn't come to rest"
                    ),
                    "objects": [bowl, tomato],
                })

            def step(self, **kwargs):
                self.calls.append(kwargs)
                success = not (
                    kwargs["action"] == "PlaceObjectAtPoint"
                    and kwargs.get("forceKinematic") is False
                )
                self.last_event = self._event(success)
                return self.last_event

        controller = Controller()
        actions = self.actions_module.Actions(controller, object())
        result = actions._restore_container_contents(
            0,
            bowl["objectId"],
            [tomato],
        )

        placements = [
            call for call in controller.calls
            if call["action"] == "PlaceObjectAtPoint"
        ]
        self.assertTrue(result.metadata["lastActionSuccess"])
        self.assertEqual(
            [call["forceKinematic"] for call in placements],
            [False, True],
        )
        self.assertEqual(placements[0]["position"], placements[1]["position"])

    def test_wash_requires_held_object_and_running_faucet(self):
        agent_event = SimpleNamespace(metadata={
            "agent": {
                "position": {"x": 0.0, "y": 0.9, "z": 0.0},
            },
            "inventoryObjects": [
                {"objectId": "Spoon|1", "objectType": "Spoon"},
            ],
            "objects": [
                {"objectId": "Faucet|1", "objectType": "Faucet", "isToggled": True},
            ],
        })
        event = SimpleNamespace(
            events=[agent_event],
            metadata=agent_event.metadata,
        )
        class Controller:
            def __init__(self, last_event):
                self.last_event = last_event

            def step(self, **_kwargs):
                for per_agent_event in self.last_event.events:
                    per_agent_event.metadata["lastActionSuccess"] = True
                self.last_event.metadata["lastActionSuccess"] = True
                return self.last_event

        controller = Controller(event)
        actions = self.actions_module.Actions(controller, object())

        self.assertIs(actions.wash("robot1", "spoon"), event)

        agent_event.metadata["objects"][0]["isToggled"] = False
        with self.assertRaisesRegex(RuntimeError, "Faucet must be on"):
            actions.wash("robot1", "spoon")


if __name__ == "__main__":
    unittest.main()

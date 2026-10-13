import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from run_ai2thor import _merge_scene_placements

ROOT = Path(__file__).resolve().parents[1]


class Map1ConfigurationTests(unittest.TestCase):
    def test_apple_and_tomato_scene_positions(self):
        scene = json.loads((ROOT/'maps_ai2thor/map1/scene.json').read_text())
        placements = {p['name']: p for p in scene['object_placements']}
        self.assertAlmostEqual(placements['Apple']['position']['z'], -.8119600415)
        self.assertEqual(placements['Tomato']['position']['x'], .33)
        self.assertEqual(placements['Tomato']['position']['z'], .88)
        self.assertTrue(placements['Apple']['apply'])
        self.assertTrue(placements['Tomato']['apply'])
        staged = scene['object_placements'][:2]
        self.assertEqual([p['name'] for p in staged], ['Tomato', 'Apple'])
        self.assertNotEqual(staged[0]['position']['x'], placements['Apple']['position']['x'])
        self.assertEqual(sum(p['name'] == 'Tomato' for p in scene['object_placements']), 1)

    def test_scene_bowl_coordinates_and_immutable_layout_merge(self):
        scene = json.loads((ROOT/'maps_ai2thor/map1/scene.json').read_text())
        layout = {'objects': [{'name': 'Apple', 'apply': False}]}
        merged = _merge_scene_placements(scene, layout)
        bowl = merged['objects'][-1]
        self.assertEqual(bowl['name'], 'Bowl')
        self.assertEqual(bowl['position'], {'x': -.19, 'z': .79})
        self.assertTrue(bowl['apply'])
        self.assertEqual(len(layout['objects']), 1)
        self.assertEqual(merged['robots'][0]['id'], 'R2')
        self.assertEqual(merged['robots'][0]['position'], {'x':-1.15,'z':1.13})
        self.assertEqual(merged['robots'][1]['id'], 'R1')
        self.assertEqual(merged['robots'][1]['position'], {'x':.67,'z':-1.61})

    def test_task_ownership_and_bread_microwave_dependencies(self):
        plan = json.loads((ROOT/'maps_ai2thor/map1/actions.json').read_text())
        tasks = {t['id']: t for t in plan['tasks']}
        ownership = {t:r for r, ids in plan['robot_plans'].items() for t in ids}
        self.assertEqual(set(ownership), set(tasks))
        self.assertEqual(plan['robot_plans'],
                         {'R1': [2,4], 'R2': [1,3,8,9,10,5], 'R3': [6,7]})
        self.assertEqual(tasks[3]['action'], 'place')
        self.assertEqual(tasks[1]['object'], 'bread')
        self.assertEqual(tasks[1]['node'], [])
        self.assertEqual(tasks[2]['node'], [])
        self.assertEqual(tasks[3]['object'], 'bread')
        self.assertEqual(tasks[3]['target'], 'microwave')
        self.assertEqual(tasks[3]['node'], [1])
        self.assertEqual(tasks[3]['wait_for_on_conflict'], ['R1'])
        self.assertEqual(tasks[4]['action'], 'place')
        self.assertEqual(tasks[4]['object'], 'apple')
        self.assertEqual(tasks[4]['node'], [2])
        self.assertEqual(tasks[4]['target'], 'sinkbasin')
        self.assertEqual(tasks[5]['action'],'pick')
        self.assertEqual(tasks[5]['object'],'bread')
        self.assertEqual(tasks[6]['action'],'pick')
        self.assertEqual(tasks[6]['object'],'tomato')
        self.assertEqual(tasks[7]['action'],'place')
        self.assertEqual(tasks[7]['object'],'tomato')
        self.assertEqual(tasks[7]['target'],'bowl')
        self.assertEqual(tasks[5]['node'],[10])
        self.assertEqual(tasks[8]['action'],'close')
        self.assertEqual(tasks[8]['node'],[3])
        self.assertEqual(tasks[9]['action'],'wait')
        self.assertEqual(tasks[9]['duration'],2.)
        self.assertEqual(tasks[9]['node'],[8])
        self.assertEqual(tasks[10]['action'],'open')
        self.assertEqual(tasks[10]['node'],[9])
        self.assertEqual(tasks[6]['node'],[])
        self.assertEqual(tasks[7]['node'],[6])
        self.assertFalse(any(t['object']=='apple' and t['target']=='bowl' for t in tasks.values()))
        self.assertFalse(any(t['object']=='fridge' for t in tasks.values()))

    def test_microwave_initial_state_is_configured_and_applied_before_navigation(self):
        scene=json.loads((ROOT/'maps_ai2thor/map1/scene.json').read_text())
        self.assertEqual(scene['initial_object_states'],[{'name':'Microwave','instance':0,'is_open':True}])
        merged=_merge_scene_placements(scene,{})
        self.assertEqual(merged['initial_object_states'],scene['initial_object_states'])
        tree=ast.parse((ROOT/'src/simulations/ai2thor/controller.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_apply_initial_object_states')
        namespace={'require_success':lambda event,operation:event}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'controller.py','exec'),namespace)
        obj={'objectId':'Microwave|1','objectType':'Microwave','openable':True,'isOpen':False}
        calls=[]
        controller=SimpleNamespace(last_event=SimpleNamespace(metadata={'objects':[obj]}),
                                   step=lambda **kw:calls.append(kw))
        namespace['_apply_initial_object_states'](controller,scene['initial_object_states'])
        self.assertEqual(calls[0]['action'],'OpenObject')
        self.assertEqual(calls[0]['objectId'],'Microwave|1')
        obj['isOpen']=True
        namespace['_apply_initial_object_states'](controller,scene['initial_object_states'])
        self.assertEqual(len(calls),1)
        obj['openable']=False
        with self.assertRaisesRegex(ValueError,'not openable'):
            namespace['_apply_initial_object_states'](controller,scene['initial_object_states'])

    def test_wait_duration_is_preserved_in_execution_and_remaining_workload(self):
        from simulations.ai2thor.execution import ExecutionPlanExecutor,build_action_sequence,task_action_duration
        plan=json.loads((ROOT/'maps_ai2thor/map1/actions.json').read_text())
        commands=build_action_sequence(plan)
        self.assertEqual(len(commands),10)
        wait=next(c for c in commands if c['action']=='wait')
        self.assertEqual(wait['duration'],2.)
        self.assertEqual(task_action_duration(wait),2.)
        executor=ExecutionPlanExecutor(plan,lambda *a,**kw:None,None)
        # Five default actions plus a two-second wait; no final place.
        bread_robot = next(r for r, ids in plan['robot_plans'].items() if 1 in ids)
        self.assertEqual(executor.estimate_remaining_action_time(bread_robot),7.)
        for duration in (-1.,float('nan')):
            with self.assertRaises(ValueError):
                task_action_duration(dict(wait,duration=duration))

    def test_object_placement_preserves_live_height_and_accepts_explicit_height(self):
        # Load only the placement function: no Unity installation/launch needed.
        tree=ast.parse((ROOT/'src/simulations/ai2thor/controller.py').read_text())
        fn=next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='_place_objects')
        namespace={'require_success': lambda event, operation: event}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'controller.py','exec'),namespace)
        calls=[]
        controller=SimpleNamespace(last_event=SimpleNamespace(metadata={'objects':[
            {'objectId':'bowl-1','objectType':'Bowl','pickupable':True,
             'position':{'x':1.,'y':.92,'z':2.}}]}),step=lambda **kwargs:calls.append(kwargs))
        namespace['_place_objects'](controller,[{'name':'Bowl','position':{'x':-.19,'z':.79}}])
        self.assertEqual(calls[-1]['position'],{'x':-.19,'y':.92,'z':.79})
        namespace['_place_objects'](controller,[{'name':'Bowl','position':{'x':-.19,'y':1.,'z':.79}}])
        self.assertEqual(calls[-1]['position']['y'],1.)

    def test_malformed_scene_positions_are_rejected(self):
        for placement in ({'name':'Bowl','position':{'x':0.}},
                          {'name':'Bowl','position':{'x':float('nan'),'z':0.}}):
            with self.assertRaises(ValueError):
                _merge_scene_placements({'object_placements':[placement]}, {})

    def test_partial_robot_placement_only_teleports_r1_and_checks_other_robots(self):
        import math
        tree=ast.parse((ROOT/'src/simulations/ai2thor/controller.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_place_agents')
        positions=[{'x':0.,'y':.9,'z':0.}, {'x':2.,'y':.9,'z':0.}, {'x':-2.,'y':.9,'z':0.}]
        goal={'x':.26,'y':.9,'z':-1.66}
        events=[SimpleNamespace(metadata={'agent':{'position':p,'rotation':{'y':0.}}}) for p in positions]
        calls=[]
        event=SimpleNamespace(metadata={'actionReturn':[goal,*positions]},events=events)
        controller=SimpleNamespace(last_event=event,step=lambda **kw:(calls.append(kw) or event))
        namespace={'AGENT_COUNT':3,'AGENT_RADIUS':.05,'_spread_agents':lambda c:None,
                   'require_success':lambda event,operation:event,
                   '_distance_xz':lambda a,b:math.hypot(a['x']-b['x'],a['z']-b['z'])}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'controller.py','exec'),namespace)
        namespace['_place_agents'](controller,[{'id':'R1','position':goal}])
        moves=[c for c in calls if c['action']=='TeleportFull']
        self.assertEqual(len(moves),1)
        self.assertEqual(moves[0]['agentId'],0)
        self.assertEqual((moves[0]['x'],moves[0]['z']),(.26,-1.66))
        with self.assertRaisesRegex(ValueError,'overlap'):
            namespace['_place_agents'](controller,[{'id':'R1','position':positions[1]}])


if __name__ == '__main__':
    unittest.main()

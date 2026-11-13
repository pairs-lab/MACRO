"""Hybrid Nav2 controller profile shared by the TurtleBot 3 (Gazebo Classic) and TurtleBot 4 (Ignition) launches."""


def add_hybrid_controllers(params):
    params['controller_server']['ros__parameters']['controller_plugins'].append('CorridorFollow')
    params['controller_server']['ros__parameters']['CorridorFollow']={
        'plugin':'nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController',
        'desired_linear_vel':.16,'lookahead_dist':.20,
        'min_lookahead_dist':.10,'max_lookahead_dist':.25,'lookahead_time':1.,
        'use_velocity_scaled_lookahead_dist':False,'transform_tolerance':.3,
        'use_rotate_to_heading':True,'rotate_to_heading_min_angle':.15,
        'rotate_to_heading_angular_vel':.25,'max_angular_accel':.8,
        'allow_reversing':False,'use_collision_detection':True,
        # At 0.16 m/s with 0.35 m/s^2 deceleration, braking is
        # ~0.037 m. Keep collision checking, but avoid the old
        # 1 s preview pre-empting the 0.25 m bound trigger.
        'max_allowed_time_to_collision_up_to_carrot':.4,
        'use_regulated_linear_velocity_scaling':True,
        'regulated_linear_scaling_min_radius':.3,
        'regulated_linear_scaling_min_speed':.04,
        'use_cost_regulated_linear_velocity_scaling':False,
        'min_approach_linear_velocity':.025,'approach_velocity_scaling_dist':.20}
    # Tight curved intervention paths need a shorter carrot than
    # the straight approach, otherwise it cuts across the corner.
    params['controller_server']['ros__parameters']['controller_plugins'].append('TurnFollow')
    params['controller_server']['ros__parameters']['TurnFollow']={
        **params['controller_server']['ros__parameters']['CorridorFollow'],
        'lookahead_dist':.06,'min_lookahead_dist':.05,'max_lookahead_dist':.10,
        'rotate_to_heading_min_angle':.5,'allow_reversing':True,
        'use_rotate_to_heading':False,'regulated_linear_scaling_min_speed':.02}
    params['controller_server']['ros__parameters']['controller_plugins'].append('ResumeFollow')
    params['controller_server']['ros__parameters']['ResumeFollow']={
        **params['controller_server']['ros__parameters']['TurnFollow'],
        'allow_reversing':False,'use_rotate_to_heading':True}

from go1_gym.envs.base.legged_robot_config import Cfg


def base_set():
    # set basics
    Cfg.terrain.teleport_robots = True
    Cfg.terrain.border_size = 50
    Cfg.terrain.num_rows = 10
    Cfg.terrain.num_cols = 10
    Cfg.commands.resampling_time = 1e9
    Cfg.env.episode_length_s = 500
    Cfg.rewards.terminal_body_height = 0.0
    Cfg.rewards.use_terminal_body_height = True


def rand_regular():
    Cfg.domain_rand.randomize_friction = True
    Cfg.domain_rand.friction_range = [0.05, 4.5]
    Cfg.domain_rand.randomize_restitution = True
    Cfg.domain_rand.restitution_range = [0, 1.0]
    Cfg.domain_rand.restitution = 0.5
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.domain_rand.added_mass_range = [-1., 3.]
    Cfg.domain_rand.randomize_com_displacement = True
    Cfg.domain_rand.com_displacement_range = [-0.1, 0.1]
    Cfg.domain_rand.randomize_motor_strength = True
    Cfg.domain_rand.motor_strength_range = [0.9, 1.1]
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.Kp_factor_range = [0.8, 1.3]
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.Kd_factor_range = [0.5, 1.5]
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.push_interval_s = 15
    Cfg.domain_rand.max_push_vel_xy = 1.


def rand_large():
    Cfg.domain_rand.randomize_friction = True
    Cfg.domain_rand.friction_range = [0.04, 6.0]
    Cfg.domain_rand.randomize_restitution = True
    Cfg.domain_rand.restitution_range = [0, 1.0]
    Cfg.domain_rand.restitution = 0.5
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.domain_rand.added_mass_range = [-1.5, 4.]
    Cfg.domain_rand.randomize_com_displacement = True
    Cfg.domain_rand.com_displacement_range = [-0.13, 0.13]
    Cfg.domain_rand.randomize_motor_strength = True
    Cfg.domain_rand.motor_strength_range = [0.88, 1.12]  # table 1 in RMA may have a typo
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.Kp_factor_range = [0.8, 1.3]
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.Kd_factor_range = [0.5, 1.5]
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.push_interval_s = 15
    Cfg.domain_rand.max_push_vel_xy = 1.


def static_low():
    Cfg.domain_rand.randomize_friction = True
    Cfg.domain_rand.friction_range = [0.05, 0.06]
    Cfg.domain_rand.randomize_restitution = True
    Cfg.domain_rand.restitution_range = [0, 0.01]
    Cfg.domain_rand.restitution = 0.5
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.domain_rand.added_mass_range = [-1., -0.99]
    Cfg.domain_rand.randomize_com_displacement = True
    Cfg.domain_rand.com_displacement_range = [-0.1, -0.09]
    Cfg.domain_rand.randomize_motor_strength = True
    Cfg.domain_rand.motor_strength_range = [0.9, -0.99]
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.Kp_factor_range = [0.8, 1.3]
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.Kd_factor_range = [0.5, 1.5]
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.push_interval_s = 15
    Cfg.domain_rand.max_push_vel_xy = 1.


def static_medium():
    Cfg.domain_rand.randomize_friction = True
    Cfg.domain_rand.friction_range = [1.0, 1.01]
    Cfg.domain_rand.randomize_restitution = True
    Cfg.domain_rand.restitution_range = [0.5, 0.51]
    Cfg.domain_rand.restitution = 0.5
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.domain_rand.added_mass_range = [0.0, 0.01]
    Cfg.domain_rand.randomize_com_displacement = True
    Cfg.domain_rand.com_displacement_range = [0.0, 0.01]
    Cfg.domain_rand.randomize_motor_strength = True
    Cfg.domain_rand.motor_strength_range = [1.0, 1.01]
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.Kp_factor_range = [0.8, 1.3]
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.Kd_factor_range = [0.5, 1.5]
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.push_interval_s = 15
    Cfg.domain_rand.max_push_vel_xy = 1.


def static_high():
    Cfg.domain_rand.randomize_friction = True
    Cfg.domain_rand.friction_range = [4.49, 4.5]
    Cfg.domain_rand.randomize_restitution = True
    Cfg.domain_rand.restitution_range = [0.99, 1.0]
    Cfg.domain_rand.restitution = 0.5
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.domain_rand.added_mass_range = [2.99, 3.]
    Cfg.domain_rand.randomize_com_displacement = True
    Cfg.domain_rand.com_displacement_range = [0.09, 0.1]
    Cfg.domain_rand.randomize_motor_strength = True
    Cfg.domain_rand.motor_strength_range = [1.09, 1.1]
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.Kp_factor_range = [0.8, 1.3]
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.Kd_factor_range = [0.5, 1.5]
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.push_interval_s = 15
    Cfg.domain_rand.max_push_vel_xy = 1.

def only_base_mass():
    Cfg.domain_rand.randomize_friction = True
    Cfg.domain_rand.friction_range = [1.0, 1.01]
    Cfg.domain_rand.randomize_restitution = True
    Cfg.domain_rand.restitution_range = [0.5, 0.51]
    Cfg.domain_rand.restitution = 0.5
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.domain_rand.added_mass_range = [-1, 3]
    Cfg.domain_rand.randomize_com_displacement = True
    Cfg.domain_rand.com_displacement_range = [0.0, 0.01]
    Cfg.domain_rand.randomize_motor_strength = True
    Cfg.domain_rand.motor_strength_range = [1.0, 1.01]
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.Kp_factor_range = [0.8, 1.3]
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.Kd_factor_range = [0.5, 1.5]
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.push_interval_s = 15
    Cfg.domain_rand.max_push_vel_xy = 1.


DR_SETTINGS = dict(
    rand_regular=rand_regular,
    rand_large=rand_large,
    static_low=static_low,
    static_medium=static_medium,
    static_high=static_high,
    only_base_mass=only_base_mass,
)

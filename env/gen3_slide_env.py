import mujoco_py
from mujoco_py import load_model_from_xml, MjSim, MjViewer, MjRenderContextOffscreen
from mujoco_py import cymj
from mujoco_py.utils import remove_empty_lines
from mujoco_py.builder import build_callback_fn
from gym import spaces
import numpy as np
from collections import deque
from env.vel_control import VelocityController

ROBOT_INIT_POS = [-0.07370902, 0.18526047, -3.05346724, -1.93002792, -0.01739147, -1.04480512, 1.59032335]
DOF = 7
TWIST_SPACE = 6
ACTION_SPACE = 3
ACTION_HIGH = 0.5

class Gen3SlideEnv:
    def __init__(self, model_path, speed=1.0, distance_threshold=0.01, reward_type='sparse', nsubsteps=20):
        with open(model_path, 'r') as f:
            self.model = mujoco_py.load_model_from_xml(f.read())
        self.sim = MjSim(self.model, nsubsteps=nsubsteps)
        self.viewer = MjViewer(self.sim)
        self.goal = self.get_target_pos()
        self.velocity_ctrl = VelocityController()
        self.action_space = spaces.Box(low=-ACTION_HIGH, high=ACTION_HIGH, shape=(ACTION_SPACE,), dtype='float32')
        self.action = np.zeros(ACTION_SPACE)
        self.twist_ee = np.zeros(TWIST_SPACE)
        self.v_tgt = np.zeros(DOF)
        self.queue = deque(maxlen=10)
        self.queue_img = deque(maxlen=10)
        self.speed = speed
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

    def reset(self):
        self.sim.reset()
        self.set_robot_pos(ROBOT_INIT_POS)
        self.close_gripper()
        self.generate_random_target()
        self.sim.step()
        return self.get_obs()

    def step(self, action):
        self.action = action
        self.v_tgt = self.get_velocity(action)
        self.sim.data.qfrc_applied[0:DOF] = self.sim.data.qfrc_bias[0:DOF]
        self.sim.data.qvel[0:DOF] = self.v_tgt[0:DOF].T
        self.sim.step()
        self.viewer.render()
        obs = self.get_obs()
        info = {
            'is_success': self.is_success(obs["achieved_goal"], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        done = bool(info['is_success'])
        return obs, reward, done, info

    def is_success(self, achieved_goal, desired_goal):
        ''' check if the robot has reached the target '''
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)
    
    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        ''' compute the reward of the robot '''
        d = self.goal_distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def get_velocity(self, action):
        ''' get the velocity of the robot '''
        self.twist_ee[0:ACTION_SPACE] = action[0:ACTION_SPACE]
        return self.speed*self.velocity_ctrl.get_joint_vel_worldframe(self.twist_ee, np.array(self.sim.data.qpos[0:DOF]), np.array(self.sim.data.qvel[0:DOF]))

    def close_gripper(self):
        ''' close the gripper '''
        gripper1_index = DOF
        gripper2_index = DOF + 1
        while self.sim.data.ctrl[gripper1_index] < 1.5 and self.sim.data.ctrl[gripper2_index] < 1.5:
            self.sim.data.ctrl[gripper1_index] = self.sim.data.ctrl[gripper1_index]+0.01
            self.sim.data.ctrl[gripper2_index] = self.sim.data.ctrl[gripper2_index]+0.01

    def generate_random_target(self):
        ''' generate a random target '''
        base_qpos = [0.475, 0.0, 0.82, 1., 0., 0., 0.]
        base_qpos[0] += np.random.uniform(-0.15, 0.15)
        base_qpos[1] += np.random.uniform(-0.15, 0.15)
        self.sim.data.set_joint_qpos('target0:joint', base_qpos)
        self.goal = self.get_target_pos()
        return self.goal

    def get_obs(self):
        ''' get the observation of the robot '''
        grip_pos = self.get_robot_grip_xpos()
        return {
            "observation": grip_pos,
            "achieved_goal": grip_pos,
            "desired_goal": np.array(self.goal),
        }

    def get_env_speed(self):
        ''' get the speed of the environment '''
        return self.speed

    def get_target_pos(self):
        ''' get the position of the target '''
        return self.sim.data.get_joint_qpos('target0:joint')[0:3]

    def get_robot_joint_names(self):
        ''' get the names of the robot's joints '''
        return self.sim.model.joint_names
    
    def get_robot_qvel_names(self):
        ''' get the names of the robot's motors '''
        return self.sim.model.actuator_names

    def get_robot_qpos(self):
        ''' get the position of the robot '''
        return self.sim.data.qpos[0:DOF]
    
    def get_robot_grip_xpos(self):
        ''' get the position of the gripper '''
        return self.sim.data.get_site_xpos('robot0:grip')

    def set_env_speed(self, speed):
        ''' set the speed of the environment '''
        self.speed = speed

    def set_robot_pos(self, pos):
        ''' set the initial position of the robot '''
        self.sim.data.qpos[0:DOF] = pos

    def set_robot_ctrl(self, ctrl):
        ''' set the motors' control of the robot '''
        self.sim.data.ctrl[0:-1] = ctrl

# # for testing purpose
# if __name__ == "__main__":
#     env = Gen3SlideEnv('gen3_slide.xml', 4, 2)
#     obs = env.reset()
#     print("observation: ", obs)
#     speed = .01
#     step = 0
#     while True:
#         if step % 10 == 0 and speed < 1 and speed > 0:
#             speed += .01
#         elif speed >= 1:
#             speed = -0.01
#         elif step % 10 == 0 and speed < 0 and speed > -1:
#             speed -= .01
#         elif speed <= -1:
#             speed = 0.01
#         print(env.step([0, 0, speed]))
#         step += 1
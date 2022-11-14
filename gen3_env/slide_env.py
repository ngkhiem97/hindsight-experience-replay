import mujoco_py
from mujoco_py import load_model_from_xml, MjSim, MjViewer, MjRenderContextOffscreen
from mujoco_py import cymj
from mujoco_py.utils import remove_empty_lines
from mujoco_py.builder import build_callback_fn
import numpy as np
from collections import deque
import threading
from control import VelocityController
from pprint import pprint

ROBOT_INIT_POS = [-0.07370902, 0.18526047, -3.05346724, -1.93002792, -0.01739147, -1.04480512, 1.59032335]
DOF = 7
TWIST_SPACE = 6

class Gen3SlideEnv:
    def __init__(self, modelFile, action_space, speed=1.0):
        with open(modelFile, 'r') as f:
            self.model = mujoco_py.load_model_from_xml(f.read())
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        self.goal = self.sim.data.get_body_xpos('Goal')
        self.velocity_ctrl = VelocityController()
        self.action = np.zeros(action_space)
        self.twist_ee = np.zeros(TWIST_SPACE)
        self.v_tgt = np.zeros(DOF)
        self.queue = deque(maxlen=10)
        self.queue_img = deque(maxlen=10)
        self.speed = speed

    def reset(self):
        self.sim.reset()
        self.set_robot_pos(ROBOT_INIT_POS)
        self.close_gripper()
        print(self.sim.data.get_joint_qpos('target0:joint'))
        self.sim.data.set_joint_qpos('target0:joint', [0.375, 0.1, 0.82, 1., 0., 0., 0.])
        return self.get_robot_grip_xpos()

    def step(self, action):
        self.action = action
        self.v_tgt = self.get_velocity(action)
        self.sim.data.qfrc_applied[0:DOF] = self.sim.data.qfrc_bias[0:DOF]
        self.sim.data.qvel[0:DOF] = self.v_tgt[0:DOF].T
        self.sim.step()
        self.viewer.render()

    def get_velocity(self, action):
        ''' get the velocity of the robot '''
        self.twist_ee[0:3] = action[0:3]
        return self.speed*self.velocity_ctrl.get_joint_vel_worldframe(self.twist_ee, np.array(self.sim.data.qpos[0:DOF]), np.array(self.sim.data.qvel[0:DOF]))

    def close_gripper(self):
        ''' close the gripper '''
        gripper1_index = DOF
        gripper2_index = DOF + 1
        while self.sim.data.ctrl[gripper1_index] < 1.5 and self.sim.data.ctrl[gripper2_index] < 1.5:
            self.sim.data.ctrl[gripper1_index] = self.sim.data.ctrl[gripper1_index]+0.01
            self.sim.data.ctrl[gripper2_index] = self.sim.data.ctrl[gripper2_index]+0.01

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

    def set_robot_pos(self, pos):
        ''' set the initial position of the robot '''
        self.sim.data.qpos[0:DOF] = pos

    def set_robot_ctrl(self, ctrl):
        ''' set the motors' control of the robot '''
        self.sim.data.ctrl[0:-1] = ctrl

if __name__ == "__main__":
    env = Gen3SlideEnv('gen3_slide.xml', 4, 4)
    env.reset()
    speed = .01
    step = 0
    while True:
        if step % 10 == 0 and speed < 1 and speed > 0:
            speed += .01
        elif speed >= 1:
            speed = -0.01
        elif step % 10 == 0 and speed < 0 and speed > -1:
            speed -= .01
        elif speed <= -1:
            speed = 0.01
        env.step([speed, 0, 0, 0])
        step += 1
        print(env.sim.data.get_site_xpos('robot0:grip'))
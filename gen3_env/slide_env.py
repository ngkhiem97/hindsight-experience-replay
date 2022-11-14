import mujoco_py
from mujoco_py import load_model_from_xml, MjSim, MjViewer, MjRenderContextOffscreen
from mujoco_py import cymj
from mujoco_py.utils import remove_empty_lines
from mujoco_py.builder import build_callback_fn
import numpy as np
from collections import deque
import threading
from control import VelocityController

ROBOT_INIT_POS = [-0.07370902, 0.18526047, -3.05346724, -1.93002792, -0.01739147, -1.04480512, 1.59032335]

class Gen3SlideEnv:
    def __init__(self, modelFile, nv):
        self.lock = threading.Lock()
        self.lock1 = threading.Lock()
        self.lockcv2 = threading.Lock()
        with open(modelFile, 'r') as f:
            self.model = mujoco_py.load_model_from_xml(f.read())
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        self.goal = self.sim.data.get_site_xpos('target0')
        self.velocityCtrl = VelocityController()
        self.nv = nv
        self.v_tgt = np.zeros(self.nv)
        self.queue = deque(maxlen=10)
        self.queue_img = deque(maxlen=10)

    def update_v(self):
        if len(self.queue) == 0:
            return
        self.lock.acquire()
        v_tgt_new = self.queue.popleft()            
        self.v_tgt = v_tgt_new
        self.lock.release()

    def reset(self):
        self.sim.reset()
        self.set_robot_pos(ROBOT_INIT_POS)
        self.close_gripper()

    def step(self):
        self.lock1.acquire()
        self.sim.data.qfrc_applied[0:self.nv] = self.sim.data.qfrc_bias[0:self.nv]
        self.sim.data.qvel[0:self.nv] = self.v_tgt
        self.sim.step()
        self.lock1.release()
        self.viewer.render()

    def close_gripper(self):
        ''' close the gripper '''
        while self.sim.data.ctrl[self.nv] < 1.5:
            self.sim.data.ctrl[self.nv] = self.sim.data.ctrl[self.nv]+0.01
            self.sim.data.ctrl[self.nv+1] = self.sim.data.ctrl[self.nv+1]+0.01

    def get_robot_joint_names(self):
        ''' get the names of the robot's joints '''
        return self.sim.model.joint_names
    
    def get_robot_qvel_names(self):
        ''' get the names of the robot's motors '''
        return self.sim.model.actuator_names

    def set_robot_pos(self, pos):
        ''' set the initial position of the robot '''
        self.sim.data.qpos[0:7] = pos

    def set_robot_ctrl(self, ctrl):
        ''' set the motors' control of the robot '''
        self.sim.data.ctrl[0:-1] = ctrl

if __name__ == "__main__":
    env = Gen3SlideEnv('gen3_slide.xml', 7)
    env.reset()
    print(env.get_robot_joint_names())
    while True:
        env.step()
from env.gen3_slide_env import Gen3SlideEnv
from rl_modules.ddpg_agent import ddpg_agent
from arguments import get_args
import os

if __name__ == "__main__":
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    args = get_args()
    env = Gen3SlideEnv('./assets/gen3_slide.xml', 8, reward_type='dense')
    obs = env.reset()
    env_params = {
        'obs': obs['observation'].shape[0],
        'goal': obs['desired_goal'].shape[0],
        'action': env.action_space.shape[0],
        'action_max': env.action_space.high[0],
        'max_timesteps': env.sim.nsubsteps
    }
    ddpg_ag = ddpg_agent(args, env, env_params)
    ddpg_ag.learn()
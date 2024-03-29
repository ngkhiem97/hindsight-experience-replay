import torch
from rl_modules.models import actor
from arguments import get_args
from env.gen3_slide_env import Gen3SlideEnv
import numpy as np

MAX_EPISODE_STEPS = 20

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

if __name__ == '__main__':
    args = get_args()
    model_path = args.save_dir + 'gen3_slide' + '/model-2023-06-28.pt'
    o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    env = Gen3SlideEnv('./assets/gen3_slide.xml', 8, reward_type='dense', viewer=True)
    observation = env.reset()
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    for i in range(args.demo_length):
        observation = env.reset()
        obs = observation['observation']
        g = observation['desired_goal']
        for t in range(MAX_EPISODE_STEPS):
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            observation_new, reward, _, info = env.step(action)
            obs = observation_new['observation']
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
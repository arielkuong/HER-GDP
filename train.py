import numpy as np
import gym
import os, sys
from arguments import get_args
from ddpg_agent import ddpg_agent
import random
import torch

def get_env_params(env, args):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    #params['reward_type'] = env._kwargs.reward_type

    print('Env observation dimension: {}'.format(params['obs']))
    print('Env goal dimension: {}'.format(params['goal']))
    print('Env action dimension: {}'.format(params['action']))
    print('Env max action value: {}'.format(params['action_max']))
    print('Env max timestep value: {}'.format(params['max_timesteps']))
    return params

def launch(args):
    # create the ddpg_agent
    env = gym.make(args.env_name)
    # set random seeds for reproduce
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # get the environment parameters
    env_params = get_env_params(env,args)

    print('Run training with seed {}'.format(args.seed))
    # create the ddpg agent to interact with the environment
    ddpg_trainer = ddpg_agent(args, env, env_params)
    ddpg_trainer.learn()

if __name__ == '__main__':
    # get the params
    args = get_args()
    launch(args)

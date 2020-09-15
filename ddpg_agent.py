import torch
import os
from datetime import datetime
import numpy as np
from models import actor, critic
from replay_buffer import replay_buffer, replay_buffer_goal_density, replay_buffer_entropy
from normalizer import normalizer
from her import her_sampler

"""
ddpg with HER

"""
class ddpg_agent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params

        # create the network
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)

        # build up the target network
        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)

        # Load the model if required
        if args.load_path != None:
            o_mean, o_std, g_mean, g_std, load_actor_model, load_critic_model = torch.load(args.load_path, map_location=lambda storage, loc: storage)
            self.actor_network.load_state_dict(load_actor_model)
            self.critic_network.load_state_dict(load_critic_model)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)

        # create the replay buffer
        if self.args.prioritization == 'goaldensity':
            self.buffer = replay_buffer_goal_density(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions_goal_density)
        elif self.args.prioritization == 'entropy':
            self.buffer = replay_buffer_entropy(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions_entropy)
        else:
            self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)


        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        # path to save the model
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = os.path.join(self.model_path, 'seed_' + str(self.args.seed))
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)


    def learn(self):
        """
        train the network

        """

        best_success_rate = -1
        success_rate_all = [self._eval_agent()]

        # start to collect samples
        for epoch in range(self.args.n_epochs):
            for cycle in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []

                #print('[{}] epoch: {}, cycle: {}, collecting trajectories'.format(datetime.now(),epoch, cycle))

                for j in range(self.args.num_rollouts_per_cycle):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_norm_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_norm_tensor)
                            action = self._action_postpro(pi)
                        # feed the actions into the environment
                        observation_new, _, _, info = self.env.step(action)
                        #self.env.render()
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new

                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)

                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)


                if self.args.prioritization == 'goaldensity' or 'entropy':
                    if (cycle % self.args.fit_interval == 0) and (not cycle == 0) or (cycle == self.args.n_cycles-1):
                        print('[{}] epoch: {}, cycle: {}, updating density model'.format(datetime.now(),epoch, cycle))
                        self.buffer.fit_density_model()
                        print('[{}] density model updated.'.format(datetime.now()))


                # store the episodes
                if self.args.prioritization == 'goaldensity':
                    self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions], self.args.rank_method)
                elif self.args.prioritization == 'entropy':
                    self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions], self.args.rank_method)
                else:
                    self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])

                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])

                #print('[{}] epoch: {}, cycle: {}, updating network'.format(datetime.now(),epoch, cycle))

                for batch in range(self.args.n_batches):

                    # train the network
                    self._update_network()

                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)

            # start to do the evaluation
            success_rate = self._eval_agent()
            success_rate_all.append(success_rate)
            np.save(self.model_path + '/eval_success_rates.npy', success_rate_all)

            if success_rate >= best_success_rate:
                best_success_rate = success_rate
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict(), self.critic_network.state_dict()], \
                            self.model_path + '/model_best.pt')

            print('[{}] epoch: {}, eval success rate: {:.3f}, best success rate: {:.3f}'.format(datetime.now(), epoch, success_rate, best_success_rate))


    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs


    def _dump_buffer(self, epoch):
        self.buffer.dump_buffer(epoch)


    # this function will choose action for the agent and do the exploration
    def _action_postpro(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])

        # generate random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])

        # choose if to use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_normal_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        if self.args.prioritization == 'goaldensity':
            transitions = self.buffer.sample(self.args.batch_size, self.args.rank_method, self.args.temperature)
        elif self.args.prioritization == 'entropy':
            transitions = self.buffer.sample(self.args.batch_size, self.args.rank_method, self.args.temperature)
        else:
            transitions = self.buffer.sample(self.args.batch_size)

        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)

        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()

        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma) #50
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)

        critic_loss = (target_q_value - real_q_value).pow(2).mean()

        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()

        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        #print('network updated')


    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        return local_success_rate

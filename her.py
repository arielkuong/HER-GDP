import numpy as np
import random

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else: # if 'replay_strategy' == 'none', do not reform transitions
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions

        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                        for key in episode_batch.keys()}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previous selected Here
        # transitions (defined by her_indexes). Keep original goal for the other
        # transitions.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # Re-compute reward since we may have substituted the goal.
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                        for k in transitions.keys()}

        return transitions

    def sample_normal_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions

        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                        for key in episode_batch.keys()}

        # Compute reward since we may have substituted the goal.
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                        for k in transitions.keys()}

        return transitions

    def sample_her_transitions_goal_density(self, episode_batch, batch_size_in_transitions, rank_method, temperature, update_stats=False):

        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions

        her_sample = np.random.uniform(size=batch_size)
        her_indexes = np.where(her_sample < self.future_p)
        non_her_indexes = np.where(her_sample >= self.future_p)

        transitions = {}
        if not update_stats:
            # Perform goal density based her sample
            p_density = episode_batch['p']
            p_density = np.power(p_density, 1/(temperature+1e-2))
            p_density = p_density / p_density.sum()
            buffer_idxs_density = np.random.choice(rollout_batch_size * T, size=batch_size, replace=True, p=p_density.flatten())
            episode_idxs = (buffer_idxs_density/T).astype(int)
            future_ts = (buffer_idxs_density%T).astype(int) + 1
            t_samples = np.random.randint(0, future_ts, size=batch_size)

            for key in episode_batch.keys():
                if not key =='p' and not key == 'pair' and not key == 'd':
                    transitions[key] = episode_batch[key][episode_idxs, t_samples].copy()

            future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_ts[her_indexes]]
            transitions['g'][her_indexes] = future_ag

            # Perform original ddpg sample for none her samples
            # select which rollouts and which timesteps to be used
            non_her_episode_idxs = (np.random.randint(0, rollout_batch_size, size=batch_size))[non_her_indexes]
            non_her_t_samples = (np.random.randint(T, size=batch_size))[non_her_indexes]
            for key in episode_batch.keys():
                if not key =='p' and not key == 'pair' and not key == 'd':
                    transitions[key][non_her_indexes] = episode_batch[key][non_her_episode_idxs, non_her_t_samples].copy()

        else:
            # Perform original her sample
            # select which rollouts and which timesteps to be used
            episode_idxs = np.random.randint(0, rollout_batch_size, size=batch_size)
            t_samples = np.random.randint(T, size=batch_size)
            # Select a future timestep in the same episode as the new goal
            future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
            future_offset = future_offset.astype(int)
            future_ts = t_samples + 1 + future_offset

            for key in episode_batch.keys():
                if not key =='p' and not key == 'pair' and not key == 'd':
                    transitions[key] = episode_batch[key][episode_idxs, t_samples].copy()

            future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_ts[her_indexes]]
            transitions['g'][her_indexes] = future_ag

        # Re-compute reward since we may have substituted the goal.
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                        for k in transitions.keys()}

        return transitions

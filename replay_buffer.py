import threading
import numpy as np

from scipy.stats import rankdata
import math

from sklearn import mixture


"""
the replay buffer here is basically from the openai baselines code

"""
class replay_buffer:
    def __init__(self, env_params, size_in_transitions, sample_func):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size_in_rollouts = size_in_transitions // self.T
        # memory management
        self.current_size_in_rollouts = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size_in_rollouts, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size_in_rollouts, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size_in_rollouts, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size_in_rollouts, self.T, self.env_params['action']]),
                        }
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]

        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['ag'][idxs] = mb_ag
            self.buffers['g'][idxs] = mb_g
            self.buffers['actions'][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            assert self.current_size_in_rollouts > 0
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size_in_rollouts]

        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]

        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)

        for key in (['r', 'obs_next', 'ag_next'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size_in_rollouts, "Batch committed to replay is too large!"

        # Increment consecutively until hit the end
        if self.current_size_in_rollouts+inc <= self.size_in_rollouts:
            idx = np.arange(self.current_size_in_rollouts, self.current_size_in_rollouts+inc)
        elif self.current_size_in_rollouts < self.size_in_rollouts:
            overflow = inc - (self.size_in_rollouts - self.current_size_in_rollouts)
            idx_a = np.arange(self.current_size_in_rollouts, self.size_in_rollouts)
            idx_b = np.random.randint(0, self.current_size_in_rollouts, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size_in_rollouts, inc)

        # Update the replay size
        self.current_size_in_rollouts = min(self.size_in_rollouts, self.current_size_in_rollouts+inc)

        if inc == 1:
            idx = idx[0]
        return idx


class replay_buffer_goal_density:
    def __init__(self, env_params, size_in_transitions, sample_func):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size_in_rollouts = size_in_transitions // self.T

        # memory management
        self.current_size_in_rollouts = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func

        self.clf = 0
        self.pred_min = 0
        self.pred_sum = 0
        self.pred_avg = 0

        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size_in_rollouts, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size_in_rollouts, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size_in_rollouts, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size_in_rollouts, self.T, self.env_params['action']]),

                        'pair':np.empty([self.size_in_rollouts * self.T, 2*self.env_params['goal']]),
                        'd': np.zeros([self.size_in_rollouts * self.T, 1]), # goal pair density
                        'p': np.zeros([self.size_in_rollouts * self.T, 1]), # priority/ranking
                        }

        # thread lock
        self.lock = threading.Lock()


    def fit_density_model(self):
        X_train = self.buffers['pair'][:(self.current_size_in_rollouts * self.T)]
        self.clf = mixture.BayesianGaussianMixture(weight_concentration_prior_type="dirichlet_distribution", n_components=10, max_iter=300)
        self.clf.fit(X_train)

        pred = self.clf.score_samples(X_train)

        self.pred_min = pred.min()
        pred = pred - self.pred_min
        pred = np.clip(pred, 0, None)
        self.pred_sum = pred.sum()
        pred = pred / self.pred_sum
        # self.pred_avg = (1 / pred.shape[0])

        with self.lock:
            self.buffers['d'][:(self.current_size_in_rollouts * self.T)] = pred.reshape(-1,1).copy()


    # store the episode
    def store_episode(self, episode_batch, rank_method):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        mb_pairs = []
        mb_d = []

        for i in range(mb_ag.shape[0]):
            for j in range(mb_ag.shape[1]-1):
                mb_pairs.append(np.concatenate([mb_ag[i][0],mb_ag[i][j+1]],axis=None))
        mb_pairs = np.array(mb_pairs)

        if not isinstance(self.clf, int):
            pred = -self.clf.score_samples(mb_pairs)

            pred = pred - self.pred_min
            pred = np.clip(pred, 0, None)
            pred = pred / self.pred_sum

            mb_d = pred.reshape(-1,1)

        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)

            # store the informations
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['ag'][idxs] = mb_ag
            self.buffers['g'][idxs] = mb_g
            self.buffers['actions'][idxs] = mb_actions

            idxs_pair = np.arange(0, batch_size)

            for i in range(self.T):
                self.buffers['pair'][idxs*self.T + i] = mb_pairs[idxs_pair*self.T + i]

            self.n_transitions_stored += self.T * batch_size

            if not isinstance(self.clf, int):
                for i in range(self.T):
                    self.buffers['d'][idxs*self.T + i] = mb_d[idxs_pair*self.T + i]

                density_pairs = self.buffers['d'][:(self.current_size_in_rollouts * self.T)]

                if rank_method == 'density':
                    complement_density = 1 - density_pairs
                    self.buffers['p'][:(self.current_size_in_rollouts * self.T)] = complement_density.copy()
                elif rank_method == 'rank':
                    complement_density = 1 - density_pairs
                    density_rank = rankdata(complement_density, method='dense')
                    density_rank = density_rank - 1
                    density_rank = density_rank.reshape(-1, 1)
                    self.buffers['p'][:(self.current_size_in_rollouts * self.T)] = density_rank.copy()

    # sample the data from the replay buffer
    def sample(self, batch_size, rank_method, temperature):
        temp_buffers = {}
        with self.lock:
            assert self.current_size_in_rollouts > 0
            for key in self.buffers.keys():
                if key == 'd' or key == 'p':
                    temp_buffers[key] = self.buffers[key][:(self.current_size_in_rollouts * self.T)]
                elif not key == 'pair':
                    temp_buffers[key] = self.buffers[key][:self.current_size_in_rollouts]

        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]

        # sample transitions
        if not isinstance(self.clf, int):
            transitions = self.sample_func(temp_buffers, batch_size, rank_method, temperature)
        else:
            transitions = self.sample_func(temp_buffers, batch_size, rank_method, temperature, update_stats = True)

        # in case the wrong sample function is used
        for key in (['r', 'obs_next', 'ag_next'] + list(self.buffers.keys())):
            if not key == 'p' and not key == 'd' and not key == 'pair':
                assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size_in_rollouts, "Batch committed to replay is too large!"

        # Increment consecutively until hit the end
        if self.current_size_in_rollouts+inc <= self.size_in_rollouts:
            idx = np.arange(self.current_size_in_rollouts, self.current_size_in_rollouts+inc)
        elif self.current_size_in_rollouts < self.size_in_rollouts:
            overflow = inc - (self.size_in_rollouts - self.current_size_in_rollouts)
            idx_a = np.arange(self.current_size_in_rollouts, self.size_in_rollouts)
            idx_b = np.random.randint(0, self.current_size_in_rollouts, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size_in_rollouts, inc)

        # Update the replay size
        self.current_size_in_rollouts = min(self.size_in_rollouts, self.current_size_in_rollouts+inc)

        if inc == 1:
            idx = idx[0]
        return idx

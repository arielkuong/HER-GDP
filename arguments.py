import argparse

"""
Here are the param for the training

"""

def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default='FetchReach-v1', help='the environment name')
    parser.add_argument('--n-epochs', type=int, default=50, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=40, help='the times to update the network')
    parser.add_argument('--num-rollouts-per-cycle', type=int, default=2, help='the rollouts per mpi')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    #parser.add_argument('--save-interval', type=int, default=1, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    #parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future',
                        help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=100, help='the number of tests')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--demo-length', type=int, default=100, help='the demo length')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--load-path', type=str, default=None, help='the path to load the previous saved models')

    parser.add_argument('--prioritization', type=str, default='goaldensity',
                        help='the prioritization strategy to be used. "goaldensity" uses GDP; "none" is vanilla HER;')

    # goal density based prioritization
    parser.add_argument('--fit-interval', type=int, default=50, help='fit interval for updating density model during a epoch, disabled when equal to --n-cycles')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature value for Entropy-Based Prioritization (MEP)')
    parser.add_argument('--rank-method', type=str, default='density', help='energy ranking method. select from "density", "rank"')

    args = parser.parse_args()

    return args

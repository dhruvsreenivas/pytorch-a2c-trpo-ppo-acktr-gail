import argparse
from multiprocessing import cpu_count
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description='Policy-optimization methods with Pytorch')
    parser.add_argument('--algo', default='ppo',
                        help='algorithm to use: a2c | ppo | acktr | trpo')
    parser.add_argument('--gail', action='store_true', default=False,
                        help='do imitation learning with gail')
    parser.add_argument('--gail-experts-dir', default='./gail_experts',
                        help='directory that contains expert demonstrations for gail')
    parser.add_argument('--gail-batch-size', type=int, default=128,
                        help='gail batch size (default: 128)')
    parser.add_argument('--gail-epoch', type=int, default=5,
                        help='gail epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='optimizer learning rate (default: 5e-5)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--damping', type=float, default=1e-3,
                        help='damping beta used to control the improvement in trpo (default=1e-3)')
    parser.add_argument('--value-l2-reg', type=float, default=1e-3,
                        help='L2 regularization for the value network (default=1e-3)')
    parser.add_argument('--max-kl', type=float, default=1e-3,
                        help='maximum kl-divergence of the original policy from the improved version in trpo'
                        ' (default=1e-3)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num-processes', type=int, default=cpu_count(),
                        help='how many training CPU processes to use '
                             '(default: number of processors available on the system)')
    parser.add_argument('--num-steps', type=int, default=20,
                        help='number of forward steps in the simulation (default: 20)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=100,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--num-env-steps', type=int, default=10e6,
                        help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--log-dir', default='./log/',
                        help='directory to save agent logs (default: ./log/)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--use-proper-time-limits', action='store_true', default=False,
                        help='compute returns taking into account time limits')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')

    # Low-precision args for TRPO/precision args in general
    parser.add_argument('--precision',
                        default='half',
                        choices=['half', 'full', 'double'],
                        help='precision for runs')
    parser.add_argument('--use-hadam', action='store_true',
                        help='use hAdam optimizer for TRPO')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr', 'trpo']
    if args.recurrent_policy:
        assert args.algo in [
            'a2c', 'ppo', 'trpo'], 'Recurrent policy is not implemented for ACKTR'

    return args

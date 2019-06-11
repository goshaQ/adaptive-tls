import sys
import ray
import subprocess
import numpy as np

from gym import spaces
from argparse import ArgumentParser

from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.dqn.dqn_policy_graph import DQNPolicyGraph
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.tune import function

from tls.agents.models import register_model
from tls.environment.sumo import SUMOEnv

_NETWORK_PATH = '/home/gosha/workspace/pycharm/adaptive-tls/networks/montgomery_county/'


def on_episode_end(info):
    env = info['env'].envs[0]  # Each worker has own list
    env.close()  # Close SUMO simulation
    # info['episode'].agent_rewards


def train(num_iters, checkpoint_freq):
    obs_space = spaces.Dict({
        'obs': spaces.Box(low=-0.5, high=1.5,
                          shape=(32, 32, 3), dtype=np.float32),
        'action_mask': spaces.Box(low=0, high=1,
                                  shape=(5,), dtype=np.int32)
    })
    act_space = spaces.Discrete(n=5)

    trainer = DQNTrainer(
        env='SUMOEnv-v0',
        config={
            'model': {
                'custom_model': 'adaptive-trafficlight',
                'custom_options': {},
            },
            'multiagent': {
                'policy_graphs': {
                    'default_policy_graph': (
                        DQNPolicyGraph,
                        obs_space,
                        act_space,
                        {},
                    ),
                },
                'policy_mapping_fn': function(lambda _: 'default_policy_graph'),
            },
            'hiddens': [],  # Don't postprocess the action scores
            'callbacks': {
                'on_episode_end': function(on_episode_end),
            },
            # 'num_workers': 4,
            # 'num_gpus_per_worker': 0.25,  # All workers on a single GPU
            'timesteps_per_iteration': 20000,
        }
    )

    for i in range(num_iters):
        print(f'== Iteration {i}==')
        print(pretty_print(trainer.train()))

        if i % checkpoint_freq == 0:
            checkpoint = trainer.save()
            print(f'\nCheckpoint saved at {checkpoint}\n')


def rollout(checkpoint_path, env='SUMOEnv-v0', steps=9999):
    subprocess.call([
        sys.executable,
        '../rollout.py', checkpoint_path,
        '--env', env,
        '--steps', str(steps),
        '--run', 'DQN',
        '--no-render',
    ])


if __name__ == '__main__':
    parser = ArgumentParser(description='Training script of Proximal Policy Optimization Agent')
    parser.add_argument('--net-file', default=_NETWORK_PATH + 'moco.net.xml',
                        help='Path to the .net.xml file')
    parser.add_argument('--config-file', default=_NETWORK_PATH + 'testmap.sumocfg',
                        help='Path to the .sumocfg file')
    parser.add_argument('--additional-file', default=_NETWORK_PATH + 'moco.det.xml',
                        help='Path to the .det.xml file')
    parser.add_argument('--num-iters', type=int, default=1000,
                        help='Number of optimization iterations')
    parser.add_argument('--checkpoint-freq', type=int, default=100,
                        help='Frequence with which a checkpoint will be created')
    parser.add_argument('--mode', choices=['train', 'eval'], default='eval',
                        help='Execution mode')
    args = parser.parse_args()

    if args.mode == 'train':
        # Register the model and environment
        register_env('SUMOEnv-v0', lambda _: SUMOEnv(net_file=args.net_file,
                                                     config_file=args.config_file,
                                                     additional_file=args.additional_file,
                                                     use_gui=True))
        register_model()

        # Initialize ray
        ray.init()

        # Train the agent
        train(args.num_iters, args.checkpoint_freq)
    elif args.mode == 'eval':
        rollout('/home/gosha/ray_results/DQN_SUMOEnv-v0_2019-06-01_00-35-2479dvwp0d/'
                'checkpoint_1/checkpoint-1')  # Should be replaced

import ray

from argparse import ArgumentParser

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.tune import function

from gym.spaces import Box, Discrete

from tls.agents.models import register_model
from tls.environment.sumo import SUMOEnv

_NETWORK_PATH = '/home/gosha/workspace/pycharm/adaptive-tls/tls/networks/montgomery_county/'


def on_episode_end(info):
    env = info['env'].envs[0]
    env.close()


def train(num_iters):
    trainer = PPOTrainer(
        env='SUMOEnv-v0',
        config={
            'model': {
                "conv_filters": [
                    [32, [4, 4], 8],
                    [64, [2, 2], 4],
                ],
            },
            'multiagent': {
                'policy_graphs': {
                    'cluster_648538736_648538737': (
                        PPOPolicyGraph,
                        Box(low=0., high=1., shape=(32, 32, 1)),
                        Discrete(n=5),
                        {}
                    ),
                    '49228579': (
                        PPOPolicyGraph,
                        Box(low=0., high=1., shape=(32, 32, 1)),
                        Discrete(n=4),
                        {}
                    ),
                    'cluster_2511020106_49297289': (
                        PPOPolicyGraph,
                        Box(low=0., high=1., shape=(32, 32, 1)),
                        Discrete(n=4),
                        {}
                    ),
                    'cluster_298135838_49135231': (
                        PPOPolicyGraph,
                        Box(low=0., high=1., shape=(32, 32, 1)),
                        Discrete(n=3),
                        {}
                    ),
                    'cluster_290051904_49145925': (
                        PPOPolicyGraph,
                        Box(low=0., high=1., shape=(32, 32, 1)),
                        Discrete(n=5),
                        {}
                    ),
                    'cluster_290051912_298136030_648538909': (
                        PPOPolicyGraph,
                        Box(low=0., high=1., shape=(32, 32, 1)),
                        Discrete(n=3),
                        {}
                    ),
                    'cluster_2511020102_2511020103_290051922_298135886': (
                        PPOPolicyGraph,
                        Box(low=0., high=1., shape=(32, 32, 1)),
                        Discrete(n=4),
                        {}
                    ),
                },
                'policy_mapping_fn': function(lambda agent_id: agent_id),
            },
            'callbacks': {
                'on_episode_end': function(on_episode_end),
            },
            # 'num_workers': 4,
            # 'num_gpus_per_worker': 0.25,  # All workers on a single GPU
            # 'timesteps_per_iteration': 16000,
        }
    )

    for i in range(num_iters):
        print(f'== Iteration {i}==')
        print(pretty_print(trainer.train()))


# TODO: Q-Mix is implemented in PyTorch!!!
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
    args = parser.parse_args()

    # Register the model and environment
    grouping = {
        'group_1': [
            'cluster_648538736_648538737',
            '49228579',
            'cluster_2511020106_49297289',
            'cluster_298135838_49135231',
            'cluster_290051904_49145925',
            'cluster_290051912_298136030_648538909',
            'cluster_2511020102_2511020103_290051922_298135886']
    }

    register_env('SUMOEnv-v0', lambda _: SUMOEnv(net_file=args.net_file,
                                                 config_file=args.config_file,
                                                 additional_file=args.additional_file,
                                                 use_gui=True))
    register_model()

    # Initialize ray
    ray.init()

    # Train the agent
    train(args.num_iters)

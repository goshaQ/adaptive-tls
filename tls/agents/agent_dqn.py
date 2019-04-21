import sys
import ray
import subprocess
import numpy as np

from gym import spaces
from ray.rllib.agents.dqn.dqn_policy_graph import DQNPolicyGraph
from ray.tune import run_experiments, function
from ray.tune.registry import register_env

from agents.models import register_model
from environment.sumo import SUMOEnv


def train():
    run_experiments({
        'agent_dqn': {
            'run': 'DQN',
            'env': 'SUMOEnv-v0',
            'resources_per_trial': {
                'cpu': 4,
            },
            'checkpoint_freq': 10,
            'config': {
                'multiagent': {
                    'policy_graphs': {
                        'cluster_298135838_49135231': (
                            DQNPolicyGraph,
                            spaces.Box(low=0.0, high=1.0, shape=(32, 32, 1), dtype=np.float32),
                            spaces.Discrete(n=3),
                            {
                                'model': {
                                    "conv_filters": [
                                        [32, [4, 4], 8],
                                        [64, [2, 2], 4]
                                    ],
                                }
                            }),
                    },
                    'policy_mapping_fn': function(lambda _: 'cluster_298135838_49135231'),
                },
            },
        },
    })


def rollout(checkpoint_path, env='SUMOEnv-v0', steps=1000):
    subprocess.call([
        sys.executable,
        '../rollout.py',
        checkpoint_path,
        '--env',
        env,
        '--steps',
        str(steps),
        '--run',
        'DQN',
        '--no-render',
    ])


if __name__ == '__main__':
    register_env('SUMOEnv-v0', lambda _: SUMOEnv(net_file='/home/gosha/Загрузки/network/moco.net.xml',
                                                 config_file='/home/gosha/Загрузки/network/testmap.sumocfg',
                                                 additional_file='/home/gosha/Загрузки/network/moco.det.xml',
                                                 use_gui=True))
    register_model()

    ray.init()
    rollout('/home/gosha/ray_results/agent_dqn/'
            'DQN_SUMOEnv-v0_0_2019-04-20_16-36-49e4a3tt46/checkpoint_50/checkpoint-50')

import sys
import ray
import subprocess
import numpy as np

from pprint import pprint

from gym import spaces
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.dqn.dqn_policy_graph import DQNPolicyGraph
from ray.tune import run_experiments, function
from ray.tune.registry import register_env

from tls.agents.models import register_model
from tls.environment.sumo import SUMOEnv


def train():
    # run_experiments({
    #     'agent_dqn': {
    #         'run': 'DQN',
    #         'env': 'SUMOEnv-v0',
    #         'resources_per_trial': {
    #             'cpu': 1,
    #             # 'gpu': 1,
    #         },
    #         'checkpoint_freq': 100,
    #         'config': {
    #             'multiagent': {
    #                 'policy_graphs': {
    #                     'cluster_298135838_49135231': (
    #                         DQNPolicyGraph,
    #                         spaces.Box(low=0.0, high=1.0, shape=(32, 32, 1), dtype=np.float32),
    #                         spaces.Discrete(n=3),
    #                         {
    #                             'model': {
    #                                 "conv_filters": [
    #                                     [32, [4, 4], 8],
    #                                     [64, [2, 2], 4]
    #                                 ],
    #                             }
    #                         }),
    #                 },
    #                 'policy_mapping_fn': function(lambda _: 'cluster_298135838_49135231'),
    #             },
    #         },
    #         'local_dir': '~/ray_results'
    #     },
    # })

    trainer = DQNTrainer(
        env='SUMOEnv-v0',
        config={
            'model': {
                'custom_model': 'adaptive-trafficlight',
                "custom_options": {},
            },
            'hiddens': [],  # Don't postprocess the action scores
            'timesteps_per_iteration': 1000,
            # 'num_workers': 4,
            # 'num_gpus_per_worker': 0.25,  # All workers on a single GPU
        }
    )

    while True:
        result = trainer.train()

        print('\n\n\n', end='')  # Delimiter
        pprint(result)


def rollout(checkpoint_path, env='SUMOEnv-v0', steps=1000):
    subprocess.call([
        sys.executable,
        '../rollout.py', checkpoint_path,
        '--env', env,
        '--steps', str(steps),
        '--run', 'DQN',
        '--no-render',
    ])


if __name__ == '__main__':
    register_env('SUMOEnv-v0', lambda _: SUMOEnv(net_file='/home/gosha/workspace/pycharm/adaptive-tls/tls/networks/montgomery_county/moco.net.xml',
                                                 config_file='/home/gosha/workspace/pycharm/adaptive-tls/tls/networks/montgomery_county/testmap.sumocfg',
                                                 additional_file='/home/gosha/workspace/pycharm/adaptive-tls/tls/networks/montgomery_county/moco.det.xml',
                                                 use_gui=True))
    register_model()

    ray.init()
    train()

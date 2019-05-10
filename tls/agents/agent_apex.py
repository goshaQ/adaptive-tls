import ray

from argparse import ArgumentParser

from ray.rllib.agents.dqn import ApexTrainer
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.tune import function

from tls.agents.models import register_model
from tls.environment.sumo import SUMOEnv

_NETWORK_PATH = '/home/gosha/workspace/pycharm/adaptive-tls/tls/networks/montgomery_county/'


def on_episode_end(info):
    env = info['env']
    env.close()  # Close SUMO simulation


def train(num_iters):
    trainer = ApexTrainer(
        env='SUMOEnv-v0',
        config={
            'model': {
                'custom_model': 'adaptive-trafficlight',
                "custom_options": {},
            },
            'hiddens': [],  # Don't postprocess the action scores
            'callbacks': {
                'on_episode_end': function(on_episode_end),
            },
            'num_workers': 8,
            'timesteps_per_iteration': 16000,
        }
    )
    for i in range(num_iters):
        print(f'== Iteration {i}==')
        print(pretty_print(trainer.train()))


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
    register_env('SUMOEnv-v0', lambda _: SUMOEnv(net_file=args.net_file,
                                                 config_file=args.config_file,
                                                 additional_file=args.additional_file,
                                                 use_gui=True))
    register_model()

    # Initialize ray
    ray.init()

    # Train the agent
    train(args.num_iters)

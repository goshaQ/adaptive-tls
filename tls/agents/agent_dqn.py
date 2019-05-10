import sys
import ray
import subprocess

from argparse import ArgumentParser


from ray.rllib.agents.dqn import DQNTrainer
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.tune import function

from tls.agents.models import register_model
from tls.environment.sumo import SUMOEnv

_NETWORK_PATH = '/home/gosha/workspace/pycharm/adaptive-tls/tls/networks/montgomery_county/'


def on_episode_end(info):
    env = info['env'].envs[0]  # Each worker has own list
    env.close()  # Close SUMO simulation
    # info['episode'].agent_rewards


def train(num_iters, checkpoint_freq):
    trainer = DQNTrainer(
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
            # 'num_workers': 4,
            # 'num_gpus_per_worker': 0.25,  # All workers on a single GPU
            'timesteps_per_iteration': 16000,
        }
    )

    for i in range(num_iters):
        print(f'== Iteration {i}==')
        print(pretty_print(trainer.train()))

        if i % checkpoint_freq:
            checkpoint = trainer.save()
            print(f'\nCheckpoint saved at {checkpoint}\n')


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
        rollout('/home/gosha/ray_results/DQN_SUMOEnv-v0_NO_TUNING_24h/')  # Should be replaced

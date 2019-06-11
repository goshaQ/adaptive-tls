import json

from time import sleep
from collections import defaultdict

from environment.sumo import SUMOEnv


def run_pretimed_simulation(_env, _n_episodes=1):
    for episode_idx in range(_n_episodes):
        obs = _env.reset()
        time = 0
        episode_reward = 0
        episode_reward_for_each = {}
        statistics = []

        done = {'__all__': False}
        while not done['__all__']:
            observation, reward, done, stat = _env.step(None)
            time += 5

            episode_reward += sum(reward.values())
            for k, v in reward.items():
                prev_reward = episode_reward_for_each.get(k, 0)
                episode_reward_for_each[k] = prev_reward + v

            # Collect statistics
            statistics.append({
                'statistics': stat.copy(),
                'episode_reward': episode_reward_for_each.copy(),
                'timestamp': time,
            })
        print(f'Episode end statistics {_env.collect_statistics_after_simulation()}')
        print(f'Episode {episode_idx} reward: {episode_reward}')
        print(f'Reward for each agent {episode_reward_for_each}')

        env.close()


def run_random_simulation(_env, _n_episodes=1):
    for episode_idx in range(_n_episodes):
        obs = _env.reset()

        done = {'__all__': False}
        while not done['__all__']:
            action = {'cluster_298135838_49135231': min(env.action_space.sample(), 2)}
            observation, reward, done, info = _env.step(action)
            sleep(1)

        env.close()


if __name__ == '__main__':
    env = SUMOEnv(net_file='/home/gosha/Загрузки/network/moco.net.xml',
                  config_file='/home/gosha/Загрузки/network/testmap.sumocfg',
                  additional_file='/home/gosha/Загрузки/network/moco.det.xml')
    run_pretimed_simulation(env)

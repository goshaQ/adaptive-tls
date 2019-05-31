from time import sleep
from environment.sumo import SUMOEnv


def run_pretimed_simulation(_env, _n_episodes=1):
    for episode_idx in range(_n_episodes):
        obs = _env.reset()
        episode_reward = 0

        done = {'__all__': False}
        while not done['__all__']:
            observation, reward, done, info = _env.step(None)
            episode_reward += sum(reward.values())

        env.close()
        print(f'Episode {episode_idx} reward: {episode_reward}')


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
    run_random_simulation(env)

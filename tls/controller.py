from environment.sumo import SUMOEnv


if __name__ == '__main__':
    env = SUMOEnv(net_file='/home/gosha/Загрузки/network/moco.net.xml',
                  config_file='/home/gosha/Загрузки/network/testmap.sumocfg',
                  additional_file='/home/gosha/Загрузки/network/moco.det.xml')

    n_episodes = 1
    for episode_idx in range(n_episodes):
        obs = env.reset()

        done = {'__all__': False}
        while not done['__all__']:
            action = {'cluster_298135838_49135231': env.action_space.sample()}
            observation, reward, done, info = env.step(action)

        env.close()

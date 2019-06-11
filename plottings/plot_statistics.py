import json
import matplotlib.pyplot as plt


def extract_statistics(_filepath):
    with open(_filepath) as f:
        tmp = json.load(f)

    # Extract reward history for an agent
    _agent_id = 'cluster_298135838_49135231'
    _timestamps = [0]
    _throughput = [0]

    for stat in tmp:
        _throughput.append(stat['episode_reward'][_agent_id])
        _timestamps.append(stat['timestamp'])

    return _timestamps, _throughput


if __name__ == '__main__':
    filepaths = [
        '/home/gosha/workspace/pycharm/adaptive-tls/plottings/pretimed_simulation_statistics.json',
        '/home/gosha/workspace/pycharm/adaptive-tls/plottings/simulation_statistics.json'
    ]

    labels = ['Fixed time', 'Adaptive', '']
    color = ['purple', 'darkorange', '']
    marker = ['o', 's', '']

    plt.figure(figsize=(7, 4))
    plt.ylim([0, 1200])
    plt.xlim([0, 7200])

    for idx, filepath in enumerate(filepaths):
        timestamps, statistics = extract_statistics(filepath)
        plt.plot(timestamps, statistics, color=color[idx], lw=1, label=labels[idx],
                 marker=marker[idx], markevery=0.1, ms=5)

    plt.ylabel('Reward (Throughput)', fontsize=11)
    plt.xlabel('Simulation time', fontsize=11)
    plt.legend(loc="lower right", prop={'size': 11})
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Reward_cluster_298135838_49135231.png')

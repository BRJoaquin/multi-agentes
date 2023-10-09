from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import product


def play_game(g, agents, num_games=1000):
    # count wins, losses, and ties
    agent_results = {
        agent: {"wins": 0, "losses": 0, "ties": 0} for agent in agents.keys()
    }
    for i in tqdm(range(num_games)):
        actions = dict(map(lambda agent: (agent, agents[agent].action()), g.agents))
        _, rewards, _, _, _ = g.step(actions)
        for agent in g.agents:
            if rewards[agent] > 0:
                agent_results[agent]["wins"] += 1
            elif rewards[agent] < 0:
                agent_results[agent]["losses"] += 1
            else:
                agent_results[agent]["ties"] += 1

    learned_strategies = dict(
        map(lambda agent: (agent, agents[agent].policy()), g.agents)
    )
    return learned_strategies, agent_results


def plot_learned_strategies(
    g, learned_strategies, agent1_name, agent2_name, agent1_id, agent2_id
):
    labels = g._moves
    agent1_strategies = learned_strategies[agent1_id]
    agent2_strategies = learned_strategies[agent2_id]

    x = range(len(labels))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(x, agent1_strategies, color="b", align="center")
    plt.xlabel("Actions")
    plt.ylabel("Probability")
    plt.title(f"{agent1_name} Learned Strategies")
    plt.xticks(x, labels)

    plt.subplot(1, 2, 2)
    plt.bar(x, agent2_strategies, color="r", align="center")
    plt.xlabel("Actions")
    plt.ylabel("Probability")
    plt.title(f"{agent2_name} Learned Strategies")
    plt.xticks(x, labels)

    plt.tight_layout()
    plt.show()


def plot_agent_results(agent_results, agent1_name, agent2_name, agent1_id, agent2_id):
    labels = ["wins", "losses", "ties"]
    agent1_results = [
        agent_results[(agent1_name, agent2_name)]["agent_results"][agent1_id][label]
        for label in labels
    ]
    agent2_results = [
        agent_results[(agent1_name, agent2_name)]["agent_results"][agent2_id][label]
        for label in labels
    ]

    x = range(len(labels))

    plt.bar(x, agent1_results, width=0.4, label=agent1_name, color="b", align="center")
    plt.bar(x, agent2_results, width=0.4, label=agent2_name, color="r", align="edge")

    plt.xlabel("Outcome")
    plt.ylabel("Count")
    plt.title(f"{agent1_name} vs {agent2_name}")
    plt.xticks(x, labels, rotation="vertical")
    plt.legend()
    plt.show()


def agent_vs_agent(g, agents, num_games=1000):
    results = {}
    for agent1_name, agent2_name in product(agents.keys(), repeat=2):
        print(f"Running experiment with {agent1_name} vs {agent2_name}")

        # Init agents
        my_agents = {}
        my_agents[g.agents[0]] = agents[agent1_name](g, g.agents[0])
        my_agents[g.agents[1]] = agents[agent2_name](g, g.agents[1])

        # Play game
        learned_strategies, agent_wins = play_game(g, my_agents, num_games)
        results[(agent1_name, agent2_name)] = {
            "learned_strategies": learned_strategies,
            "agent_results": agent_wins,
        }

        plot_agent_results(results, agent1_name, agent2_name, g.agents[0], g.agents[1])
        plot_learned_strategies(
            g, learned_strategies, agent1_name, agent2_name, g.agents[0], g.agents[1]
        )
    return results

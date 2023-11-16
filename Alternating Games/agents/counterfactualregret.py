import numpy as np
from numpy import ndarray
from base.game import AlternatingGame, AgentID, ObsType
from base.agent import Agent
from tqdm import tqdm

class Node():
    def __init__(self, game: AlternatingGame, agent: AgentID, obs: ObsType) -> None:
        self.game = game
        self.agent = agent
        self.obs = obs
        self.num_actions = game.num_actions(agent)
        self.regret_sum = np.zeros(self.num_actions)
        self.strategy_sum = np.zeros(self.num_actions)
        self.strategy = np.ones(self.num_actions) / self.num_actions

    def update_strategy(self):
        self.strategy = np.maximum(self.regret_sum, 0)
        normalizing_sum = np.sum(self.strategy)
        if normalizing_sum > 0:
            self.strategy /= normalizing_sum
        else:
            self.strategy = np.ones(self.num_actions) / self.num_actions
        self.strategy_sum += self.strategy

    def get_average_strategy(self):
        normalizing_sum = np.sum(self.strategy_sum)
        if normalizing_sum > 0:
            return self.strategy_sum / normalizing_sum
        else:
            return np.ones(self.num_actions) / self.num_actions

    def update(self, utility, node_utility, probability) -> None:
        regret = utility - node_utility
        self.regret_sum += probability * regret
        self.update_strategy()

    def policy(self):
        # Devuelve la estrategia actual del nodo
        # La estrategia se basa en los regrets positivos acumulados para cada acción
        regrets = np.maximum(self.regret_sum, 0)
        normalizing_sum = np.sum(regrets)
        if normalizing_sum > 0:
            return regrets / normalizing_sum
        else:
            # Si todos los regrets son negativos o cero, se devuelve una distribución uniforme
            return np.ones(self.num_actions) / self.num_actions

class CounterFactualRegret(Agent):
    def __init__(self, game: AlternatingGame, agent: AgentID) -> None:
        super().__init__(game, agent)
        self.node_dict: dict[ObsType, Node] = {}

    def action(self):
        try:
            node = self.node_dict[self.game.observe(self.agent)]
            return np.argmax(np.random.multinomial(1, node.get_average_strategy(), size=1))
        except KeyError:
            raise ValueError('Train agent before calling action()')
    
    def train(self, niter=1000):
        for _ in tqdm(range(niter)):
            self.cfr()

    def cfr(self):
        for agent in self.game.agents:
            self.game.reset()
            probability = np.ones(len(self.game.agents))
            self.cfr_rec(game=self.game.clone(), agent=agent, probability=probability)

    def cfr_rec(self, game: AlternatingGame, agent: AgentID, probability: ndarray):
        current_agent = game.agent_selection
        current_agent_index = self.game.agent_name_mapping[current_agent] 

        if game.done() or game.terminated():
            return game.reward(agent)

        obs = tuple(game.observe(current_agent).flatten())
        node = self.node_dict.get(obs)
        self.node_dict[obs] = node

        strategy = node.get_average_strategy()
        utility = np.zeros(game.num_actions(current_agent))
        node_utility = 0.0

        actions = game.available_actions()
        np.random.shuffle(actions)

        # Iterar sobre las acciones como en MiniMax
        for a in actions:
            next_game = game.clone()
            next_game.step(a)
            next_probability = probability.copy()
            next_probability[current_agent_index] *= strategy[a]
            utility[a] = self.cfr_rec(game=next_game, agent=agent, probability=next_probability)
            node_utility += strategy[a] * utility[a]

        if current_agent == agent:
            node.update(utility=utility, node_utility=node_utility, probability=probability[current_agent_index])

        return node_utility

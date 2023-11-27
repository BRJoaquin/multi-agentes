import numpy as np
from numpy import ndarray
from base.game import AlternatingGame, AgentID, ObsType
from base.agent import Agent
from tqdm import tqdm


class LinealNode:
    def __init__(self, game: AlternatingGame, agent: AgentID, obs: ObsType) -> None:
        """
        Initializes the CounterfactualRegret LinealNode object.

        Args:
            game (AlternatingGame): The game object representing the game being played.
            agent (AgentID): The ID of current agent.
            obs (ObsType): The observation received by the geme.
        """
        self.game = game.clone()
        self.agent = agent
        self.obs = obs

        # initialize the cumulative regrets and the learned policy
        self.cum_regret = np.zeros(game.num_actions(agent))
        self.curr_policy = np.full(
            self.game.num_actions(self.agent), 1 / self.game.num_actions(self.agent)
        )
        self.sum_policy = self.curr_policy.copy()
        self.learned_policy = self.curr_policy.copy()
        self.niter = 1

    def update(self, utility, node_utility, probability, weight) -> None:
        """
        Updates the counterfactual regret values for the current strategy profile.

        Args:
            utility: The utility value of the current strategy profile.
            node_utility: The utility value of the current node.
            probability: The probability of reaching this node for the other agents.
        """
        regret = utility - node_utility
        self.cum_regret += weight * probability * regret
        self.update_strategy()

    def policy(self):
        return self.learned_policy

    def update_strategy(self):
        """
        Updates the current strategy profile based on the counterfactual regrets.
        """
        positive_regrets = np.maximum(self.cum_regret, 0)
        sum_positive_regrets = np.sum(positive_regrets)
        if sum_positive_regrets > 0:
            self.curr_policy = positive_regrets / sum_positive_regrets
        else:
            self.curr_policy = np.full(
                self.game.num_actions(self.agent), 1 / self.game.num_actions(self.agent)
            )
        self.sum_policy += self.curr_policy
        self.niter += 1
        self.learned_policy = self.sum_policy / self.niter


class CounterFactualRegretPlus(Agent):
    def __init__(self, game: AlternatingGame, agent: AgentID) -> None:
        super().__init__(game, agent)
        self.node_dict: dict[ObsType, LinealNode] = {}

    def action(self):
        try:
            node = self.node_dict[self.game.observe(self.agent)]
            a = np.argmax(np.random.multinomial(1, node.policy(), size=1))
            return a
        except:
            raise ValueError("Train agent before calling action()")

    def train(self, niter=1000, callback=None, callback_every=100):
        for i in tqdm(range(niter)):
            weight = 1.0 / (i + 1)  # weight of the current strategy profile
            self.cfr(weight)
            if callback is not None and i % callback_every == 0:
                callback(self, i)

    def cfr(self, weight):
        game = self.game.clone()
        for agent in self.game.agents:
            game.reset()
            probability = np.ones(game.num_agents)
            self.cfr_rec(game=game, agent=agent, probability=probability, weight=weight)

    def cfr_rec(
        self, game: AlternatingGame, agent: AgentID, probability: ndarray, weight: float
    ):
        """
        Performs counterfactual regret minimization (CFR) recursively.

        Args:
            game (AlternatingGame): The game being played.
            agent (AgentID): The ID of the agent that is building the strategy.
            probability (ndarray): Vector of size num_agents, that contains the probability of each agent playing at the current node.

        Returns:
            float: The utility of the current node for the specified agent.
        """

        # get current agent
        current_agent = game.agent_selection

        # base case: game is done or terminated then return reward
        if game.done() or game.terminated():
            return game.reward(agent)

        # if current_agent.is_chance():
        #     game_clone = game.clone()
        #     action = game_clone.sample_chance(current_agent)
        #     game_clone.step(action)
        #     return self.cfr_rec(game=game_clone, agent=agent, probability=probability)

        # get observation node
        obs = game.observe(current_agent)
        try:
            node = self.node_dict[obs]
        except:
            # in case node does not exist, create it
            node = LinealNode(game=game, agent=current_agent, obs=obs)
            self.node_dict[obs] = node

        assert current_agent == node.agent

        # the utility of the current node is the expected utility of all possible actions
        utility = np.zeros(game.num_actions(current_agent))
        for a in game.action_iter(current_agent):
            # compute probability of reaching this node
            node_probability = probability.copy()
            node_probability[
                game.agent_name_mapping[current_agent]
            ] *= node.curr_policy[a]
            # play action a
            game_clone = game.clone()
            game_clone.step(a)

            utility[a] = self.cfr_rec(
                game=game_clone, agent=agent, probability=node_probability
            )

        node_utility = np.sum(utility * node.curr_policy)

        # update node cumulative regrets using regret matching
        # we only update the regrets of the agent that is building the strategy
        if current_agent == agent:
            # calculate probability of reaching this node for the other agents
            oponent_probability = probability.copy()
            oponent_probability[game.agent_name_mapping[current_agent]] = 1
            oponent_probability = np.prod(oponent_probability)

            node.update(
                utility=utility,
                node_utility=node_utility,
                probability=oponent_probability,
                weight=weight,
            )

        return node_utility

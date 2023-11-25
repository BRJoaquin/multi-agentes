from agents.counterfactualregret import CounterFactualRegret
import numpy as np
from numpy import ndarray
from base.game import AlternatingGame, AgentID, ObsType
from base.agent import Agent
from typing import Callable


class EnhancedCounterFactualRegret(CounterFactualRegret):
    def __init__(
        self,
        game: AlternatingGame,
        agent: AgentID,
        value_estimator: Callable[[AlternatingGame, AgentID], float] = None,
        max_depth=float("inf"),
    ) -> None:
        super().__init__(game, agent)
        self.value_estimator = value_estimator
        self.max_depth = max_depth
        self.current_depth = 0

    def cfr_rec(self, game: AlternatingGame, agent: AgentID, probability: ndarray):
        """
        Extends the CFR recursion with value estimation and depth limitation.

        Args:
            game (AlternatingGame): The game being played.
            agent (AgentID): The ID of the agent that is building the strategy.
            probability (ndarray): Vector of size num_agents, that contains the probability of each agent playing at the current node.
            depth (int): Current depth of the tree.

        Returns:
            float: The utility of the current node for the specified agent.
        """
        if self.current_depth >= self.max_depth:
            return self.estimate_value(game, agent)

        self.current_depth += 1
        # Call the original cfr_rec method for further recursion
        utility = super().cfr_rec(game, agent, probability)
        self.current_depth -= 1  # kind of backtracking
        return utility

    def estimate_value(self, game: AlternatingGame, agent: AgentID):
        """
        Estimate the value of a node using the provided value estimator function.

        Args:
            game (AlternatingGame): The game being played.
            agent (AgentID): The ID of the agent.

        Returns:
            float: Estimated value of the current node.
        """
        return self.value_estimator(game, agent) if self.value_estimator else 0

    def action(self):
        try:
            node = self.node_dict[self.game.observe(self.agent)]
            a = np.argmax(np.random.multinomial(1, node.policy(), size=1))
            return a
        except:
            return self.estimate_value(self.game, self.agent)

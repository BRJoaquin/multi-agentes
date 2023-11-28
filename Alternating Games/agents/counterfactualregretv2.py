from agents.counterfactualregret import CounterFactualRegret
import numpy as np
from numpy import ndarray
from base.game import AlternatingGame, AgentID, ActionType
from base.agent import Agent
from typing import Callable


class EnhancedCounterFactualRegret(CounterFactualRegret):
    def __init__(
        self,
        game: AlternatingGame,
        agent: AgentID,
        value_estimator: Callable[[AlternatingGame, AgentID], float],
        max_depth=float("inf"),
        action_selection: Callable[[AlternatingGame, AgentID], ActionType] = None,
    ) -> None:
        super().__init__(game, agent)
        self.value_estimator = value_estimator
        self.action_selection = action_selection
        self.max_depth = max_depth
        self.current_depth = 0

    def cfr_rec(self, game: AlternatingGame, agent: AgentID, probability: ndarray):
        if self.current_depth >= self.max_depth:
            return self.estimate_value(game, agent)

        self.current_depth += 1
        # Call the original cfr_rec method for further recursion
        utility = super().cfr_rec(game, agent, probability)
        self.current_depth -= 1  # kind of backtracking
        return utility

    def estimate_value(self, game: AlternatingGame, agent: AgentID):
        return self.value_estimator(game, agent)

    def action(self):
        try:
            node = self.node_dict[self.game.observe(self.agent)]
            a = np.argmax(np.random.multinomial(1, node.policy(), size=1))
        except:
            if self.action_selection is None:
                a = np.random.choice(self.game.available_actions())
            else:
                a = self.action_selection(self.game, self.agent)
        return a

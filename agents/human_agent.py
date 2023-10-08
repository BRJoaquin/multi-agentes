from itertools import product
import numpy as np
from numpy import ndarray
from base.agent import Agent
from base.game import SimultaneousGame, AgentID


class HumanAgent(Agent):
    def __init__(self, game: SimultaneousGame, agent: AgentID):
        super().__init__(game, agent)
        num_actions = self.game.num_actions(self.agent)
        self._policy = np.full(num_actions, 1 / num_actions)
        self._action_count = np.zeros(num_actions)
        self._moves_names = (
            self.game._moves
            if hasattr(self.game, "_moves")
            else list(range(num_actions))
        )
        self._moves = list(range(num_actions))

    def action(self):
        print(
            f"Select Action: {self._moves_names} by pressing [0-{len(self._moves)-1}]"
        )
        action = int(input(f"Action input [0-{len(self._moves)-1}: "))
        while action not in self._moves:
            print("Invalid action. Try again.")
            action = int(input(f"Action input [0-{len(self._moves)-1}]: "))
        self._action_count[action] += 1
        return action

    def update(self):
        self._policy = self._action_count / np.sum(self._action_count)

    def policy(self):
        self.update()
        return self._policy

import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID, ActionDict


class RegretMatching(Agent):
    def __init__(
        self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None
    ) -> None:
        super().__init__(game=game, agent=agent)
        if initial is None:
            self.curr_policy = np.full(
                self.game.num_actions(self.agent), 1 / self.game.num_actions(self.agent)
            )
        else:
            self.curr_policy = initial.copy()
        self.cum_regrets = np.zeros(self.game.num_actions(self.agent))
        self.sum_policy = self.curr_policy.copy()
        self.learned_policy = self.curr_policy.copy()
        self.niter = 1
        np.random.seed(seed=seed)

    def regrets(self, played_actions: ActionDict) -> dict[AgentID, float]:
        actions = played_actions.copy()
        a = actions[self.agent]
        u = np.zeros(self.game.num_actions(self.agent), dtype=float)
        for action in range(self.game.num_actions(self.agent)):
            g = self.game.clone()
            actions[self.agent] = action
            g.step(actions)
            u[action] = g.reward(self.agent)

        regrets = u - u[a]
        return regrets

    def regret_matching(self):
        positive_regrets = np.maximum(self.cum_regrets, 0)
        sum_positive_regrets = np.sum(positive_regrets)
        if sum_positive_regrets > 0:
            self.curr_policy = positive_regrets / sum_positive_regrets
        else:
            self.curr_policy = np.full(
                self.game.num_actions(self.agent), 1 / self.game.num_actions(self.agent)
            )
        self.sum_policy += self.curr_policy

    def update(self) -> None:
        actions = self.game.observe(self.agent)
        if actions is None:
            return
        regrets = self.regrets(actions)
        self.cum_regrets += regrets
        self.regret_matching()
        self.niter += 1
        self.learned_policy = self.sum_policy / self.niter

    def action(self):
        self.update()
        return np.argmax(np.random.multinomial(1, self.curr_policy, size=1))

    def policy(self):
        return self.learned_policy

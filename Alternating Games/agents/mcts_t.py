from base.game import AlternatingGame, AgentID, ActionType
from base.agent import Agent
from math import log, sqrt
import numpy as np
from typing import Callable


class MCTSNode:
    def __init__(self, parent: "MCTSNode", game: AlternatingGame, action: ActionType):
        self.parent = parent
        self.game = game
        self.action = action
        self.children = []
        self.explored_children = 0
        self.visits = 0
        # self.value = 0
        self.cum_rewards = np.zeros(len(game.agents))
        self.agent = game.agent_selection

    def update(self, rewards) -> None:
        self.visits += 1
        for i in range(len(rewards)):
            self.cum_rewards[i] += rewards[i]
        # self.value = (
        #     self.cum_rewards[self.game.agent_name_mapping[self.agent]] / self.visits
        # )

    def value(self, agent: AgentID) -> float:
        if self.visits == 0:
            return 0
        return self.cum_rewards[self.game.agent_name_mapping[agent]] / self.visits


def ucb(node: MCTSNode, agent: AgentID, C=sqrt(2)) -> float:
    return node.value(agent) + C * sqrt(log(node.parent.visits) / node.visits)


def uct(node: MCTSNode, agent: AgentID) -> MCTSNode:
    child = max(node.children, key=lambda child: ucb(child, agent))
    return child


class MonteCarloTreeSearch(Agent):
    def __init__(
        self,
        game: AlternatingGame,
        agent: AgentID,
        simulations: int = 100,
        rollouts: int = 10,
        selection: Callable[[MCTSNode, AgentID], MCTSNode] = uct,
    ) -> None:
        """
        Parameters:
            game: alternating game associated with the agent
            agent: agent id of the agent in the game
            simulations: number of MCTS simulations (default: 100)
            rollouts: number of MC rollouts (default: 10)
            selection: tree search policy (default: uct)
        """
        super().__init__(game=game, agent=agent)
        self.simulations = simulations
        self.rollouts = rollouts
        self.selection = selection

    def action(self) -> ActionType:
        a, _ = self.mcts()
        return a

    def estimate(self) -> float:
        _, v = self.mcts()
        return v

    def mcts(self) -> (ActionType, float):
        root = MCTSNode(parent=None, game=self.game, action=None)

        for _ in range(self.simulations):
            node = root
            node.game = self.game.clone()

            # selection
            node = self.select_node(node=node)

            # expansion
            self.expand_node(node)

            # rollout
            rewards = self.rollout(node)

            # update values / Backprop
            self.backprop(node, rewards)

        action = self.action_selection(root)

        return action, root.value(self.agent)

    def backprop(self, node, rewards):
        # cumulate rewards and visits from node to root navigating backwards through parent
        if node is None:
            return
        else:
            node.update(rewards)
            self.backprop(node.parent, rewards)

    def rollout(self, node):
        rewards = np.zeros(len(self.game.agents))

        # in case the game is already done, we don't need to rollout
        if node.game.done() or node.game.terminated():
            game_reward = node.game.rewards
            for _agent in node.game.agents:
                rewards[self.game.agent_name_mapping[_agent]] += game_reward[_agent]
            return rewards

        # we play the game until it is done (random actions)
        for _ in range(self.rollouts):
            game = node.game.clone()
            while not game.done() and not game.terminated():
                actions = game.num_actions(game.agent_selection)
                action = np.random.randint(actions)
                game.step(action)
            game_reward = game.rewards
            for _agent in game.agents:
                rewards[self.game.agent_name_mapping[_agent]] += game_reward[_agent]
        rewards /= self.rollouts
        return rewards

    def select_node(self, node: MCTSNode) -> MCTSNode:
        curr_node = node
        while curr_node.children:
            if curr_node.explored_children < len(curr_node.children):
                unexplored_children = curr_node.children[curr_node.explored_children]
                curr_node.explored_children += 1
                return unexplored_children
            else:
                curr_node = self.selection(curr_node, curr_node.agent)
                pass
        return curr_node

    def expand_node(self, node) -> None:
        if node.game.done() or node.game.terminated():
            return
        else:
            actions = node.game.num_actions(node.agent)
            for action in range(actions):
                game_clone = node.game.clone()
                game_clone.step(action)
                node.children.append(
                    MCTSNode(parent=node, game=game_clone, action=action)
                )
            return

    def action_selection(self, node: MCTSNode) -> (ActionType, float):
        best_action = None
        best_value = -float("inf")

        for child_idx in range(node.explored_children):
            child = node.children[child_idx]
            child_value = child.value(self.agent)

            if child_value > best_value:
                best_value = child_value
                best_action = child.action

        return best_action

    def print_tree(self, node: MCTSNode, indent: int = 0):
        print(
            "  " * indent,
            node.game.observe(node.agent),
            node.value(self.agent),
            node.visits,
        )
        for child in node.children:
            self.print_tree(child, indent + 1)

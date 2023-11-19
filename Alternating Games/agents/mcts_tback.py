from base.game import AlternatingGame, AgentID, ActionType, ObsType
from base.agent import Agent
from math import log, sqrt
import numpy as np
import random


class MCTSNode:
    def __init__(
        self,
        parent: "MCTSNode",
        game: AlternatingGame,
        hallucination_count=10,
    ) -> None:
        self.parent = parent  # parent node
        self.game = game.clone()  # game state at this node
        self.current_agent = self.game.agent_selection  # agent turn at this node
        self.obs = self.game.observe(self.current_agent)  # observation at this node

        # children is dictionary of action:node
        # create null children for every action in the game
        self.children = {
            action: None for action in range(game.num_actions(self.current_agent))
        }

        self.visits = 1
        self.cum_rewards = np.zeros(game.num_agents)

        if game.done() or game.terminated():
            self.is_terminal = True
            for _agent in game.agents:
                self.cum_rewards[self.game.agent_name_mapping[_agent]] = game.reward(
                    _agent
                )
        else:
            self.is_terminal = False
            for _ in range(hallucination_count):
                hallucination_rewards = self.hallucinate(self.game)
                for _agent in game.agents:
                    self.cum_rewards[
                        self.game.agent_name_mapping[_agent]
                    ] += hallucination_rewards[_agent]

            self.cum_rewards /= hallucination_count

        backpropagate(self, self.cum_rewards)

    def get_children_rewards(self):
        # return a list of the rewards for each child (action)
        return [
            child.cum_rewards[self.current_agent] / child.visits
            if child is not None
            else 0  # TODO: check if this is correct
            for child in self.children.values()
        ]

    def hallucinate(self, game: AlternatingGame):
        """
        Hallucinates the reward for the current node.

        Args:
            game (AlternatingGame): The game object representing the game being played.

        Returns:
            reward (float): The hallucinated reward for the current node.
        """
        if game.done() or game.terminated():
            return game.rewards
        else:
            # take a random action
            action = np.random.choice(game.num_actions(game.agent_selection))
            game_clone = game.clone()
            game_clone.step(action)
            return self.hallucinate(game_clone)


def backpropagate(node: MCTSNode, reward):
    if node is None:
        return
    else:
        node.visits += 1
        for agent_index in range(len(reward)):
            node.cum_rewards[agent_index] += reward[agent_index]
        backpropagate(node.parent, reward)


def ucb(node, agent: AgentID, C=sqrt(2)) -> float:
    if node is None:
        return float("inf")
    return (
        node.cum_rewards[node.game.agent_name_mapping[agent]] / node.visits
    ) + C * sqrt(log(node.parent.visits) / node.visits)


def uct(node: MCTSNode) -> MCTSNode:
    best_action = None
    best_ucb = float("-inf")
    for action, child in node.children.items():
        if ucb(child, node.current_agent) > best_ucb:
            best_ucb = ucb(child, node.current_agent)
            best_action = action
    return best_action


class MonteCarloTreeSearch(Agent):
    def __init__(
        self,
        game: AlternatingGame,
        agent: AgentID,
        simulations: int = 100,
        rollouts: int = 10,
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
        self.node_dict: dict[ObsType, MCTSNode] = {}
        self.agent = agent
        self.game = game.clone()

        self.build_tree()

    def action(self) -> ActionType:
        try:
            node = self.node_dict[self.game.observe(self.agent)]
            return np.argmax(node.get_children_rewards())
        except:
            return np.random.choice(self.game.num_actions(self.agent))

    def selection(self, node: MCTSNode) -> ActionType:
        """
        Selects the action with the highest UCT value.

        Args:
            node (MCTSNode): The node to select from.

        Returns:
            action (ActionType): The action with the highest UCT value.
        """
        # if some action is not explored, return it
        for action in node.children:
            if node.children[action] is None:
                return action
        # else return the action with the highest UCT value
        return uct(node)

    def build_tree(self):
        for _ in range(self.simulations):
            game = self.game.clone()
            game.reset()
            current_agent = game.agent_selection
            if self.node_dict.get(game.observe(current_agent)) is None:
                self.node_dict[game.observe(current_agent)] = MCTSNode(
                    parent=None,
                    game=game,
                    hallucination_count=self.rollouts,
                )
            current = self.node_dict[game.observe(current_agent)]
            action = self.selection(current)
            game.step(action)
            continue_for_loop = False
            # selection
            while current.children[action] is not None:
                current = current.children[action]
                if current.is_terminal:
                    reward_aux = np.zeros(game.num_agents)
                    game_reward = game.rewards
                    for _agent in game.agents:
                        reward_aux[self.game.agent_name_mapping[_agent]] = game_reward[
                            _agent
                        ]
                    backpropagate(current, reward_aux)
                    continue_for_loop = True
                    break
                action = self.selection(current)
                game.step(action)

            if continue_for_loop:
                continue

            # expand
            current.children[action] = MCTSNode(
                parent=current,
                game=game,
                hallucination_count=self.rollouts,
            )
            self.node_dict[game.observe(game.agent_selection)] = current.children[
                action
            ]

    def print_tree(self, current: MCTSNode, depth=0):
        if current is None:
            return
        else:
            print(
                "  " * depth,
                current.cum_rewards / current.visits,
                current.visits,
                current.current_agent,
                current.obs,
            )
            for child in current.children.values():
                self.print_tree(child, depth + 1)

from itertools import product
import numpy as np
from numpy import ndarray
from base.agent import Agent
from base.game import SimultaneousGame, AgentID


class FictitiousPlay(Agent):
    def __init__(
        self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None
    ) -> None:
        super().__init__(game=game, agent=agent)
        np.random.seed(seed=seed)

        self.count: dict[AgentID, ndarray] = {}
        self.learned_policy: dict[AgentID, ndarray] = {}

        if initial is None:
            self.count = {
                agent: np.zeros(self.game.num_actions(agent))
                for agent in self.game.agents
            }
            self.learned_policy = {
                agent: np.zeros(self.game.num_actions(agent))
                for agent in self.game.agents
            }
        else:
            self.count = initial
            self.update()

    # get_rewards retorna un diccionario de tuplas de acciones y recompensas
    # es decir, rewards[(a1, a2)] = r
    # dichas recompensas son desde el punto de vista del agente self.agent
    # para ello se clona el juego y se ejecuta step con las acciones de los agentes
    # es una idea de juego "ficticio" (simulado)
    def get_rewards(self) -> dict:
        g = self.game.clone()
        agents_actions = list(map(lambda agent: list(g.action_iter(agent)), g.agents))
        rewards: dict[tuple, float] = {}

        # Ayuda: usar product(*agents_actions) de itertools para iterar sobre agents_actions

        for actions in product(*agents_actions):
            # me clono el juego para simular las jugadas
            g = self.game.clone()
            g.step(dict(zip(g.agents, actions)))
            rewards[actions] = g.reward(self.agent)

        return rewards

    # retorna un vector de utilidades (valor) para cada acción de agente
    # es decir, utility[a] = u
    # dichas utilidades son desde el punto de vista del agente self.agent
    def get_utility(self):
        rewards = self.get_rewards()
        utility = np.zeros(self.game.num_actions(self.agent))
        #
        # TODO: calcular la utilidad (valor) de cada acción de agente. (matriz de recompensa)
        # Ayuda: iterar sobre rewards para cada acción de agente
        #

        # recorro las acciones posibles del agente
        for action in range(self.game.num_actions(self.agent)):
            # recorro las diferentes combinaciones (jugadas) de acciones de mi jugador y los otros
            for jugada in rewards.keys():
                proba = 1
                # si la accion que estoy evaluando esta dentro de la jugada
                if jugada[self.game.agent_name_mapping[self.agent]] == action:
                    # veo que jugaron los de mas, es decir recorro la jugada
                    for other_agent in range(len(jugada)):
                        # me interesan las acciones de los otros agentes
                        if other_agent != self.game.agent_name_mapping[self.agent]:
                            # obtengo la accion del otro agente
                            other_agent_action = jugada[other_agent]
                            # obtengo el id del otro agente
                            other_agent_id = self.game.agents[other_agent]
                            # actualizo la utilidad en base a la recompensa de la jugada y la politica aprendida
                            # la politica aprendida es la probabilidad de que el otro agente juegue esa accion
                            proba *= self.learned_policy[other_agent_id][
                                other_agent_action
                            ]
                    utility[action] += rewards[jugada] * proba

        return utility

    def bestresponse(self):
        # TODO: retornar la acción de mayor utilidad
        # argmax del valor de la utilidad
        return np.argmax(self.get_utility())

    def update(self) -> None:
        actions = self.game.observe(self.agent)
        if actions is None:
            return
        for agent in self.game.agents:
            self.count[agent][actions[agent]] += 1
            self.learned_policy[agent] = self.count[agent] / np.sum(self.count[agent])

    def action(self):
        self.update()
        return self.bestresponse()

    def policy(self):
        return self.learned_policy[self.agent]

# Experimento de Juegos Multiagente

## Introducción

Este informe presenta un experimento de juegos multiagente utilizando tres tipos de agentes:

1. **RandomAgent**: Este agente toma decisiones aleatorias. Se puede inicializar con una distribución específica para experimentar con estrategias sesgadas o puras.
2. **FictitiousPlay (FP)**: Este agente utiliza el algoritmo de Fictitious Play para aprender la mejor respuesta a la estrategia empírica del oponente.
3. **RegretMatching (RM)**: Este agente utiliza el algoritmo de Regret Matching para adaptar su estrategia en función del arrepentimiento acumulado.

Adicionalmente, se implementó un **HumanAgent** para permitir la interacción humana en los experimentos.

Utilizamos el framework PettingZoo y experimentamos con tres juegos: Piedra, Papel o Tijera (RPS), Matching Pennies (MP) y Blotto. Los juegos ya fueron provistos, por lo que no se detallan sus implementaciones aquí.

## Metodología

Para realizar los experimentos, simplemente definimos los agentes y los dejamos jugar un número determinado de rondas. A continuación se muestra un ejemplo de cómo se configura un experimento con el juego RPS.

```python
agents = {
    'RegretMatching': lambda env, agent_name: RegretMatching(env, agent_name),
    'RandomAgent': lambda env, agent_name: RandomAgent(env, agent_name),
    'FictitiousPlay': lambda env, agent_name: FictitiousPlay(env, agent_name),
    'RandomAgentBias': lambda env, agent_name: RandomAgent(env, agent_name, initial=[0.2, 0.2, 0.6]),
    'AlwaysRock': lambda env, agent_name: RandomAgent(env, agent_name, initial=[1, 0, 0]),
}

agent_vs_agent(g, agents, 10_000)
```

## Funciones Auxiliares
Las funciones auxiliares incluyen `play_game` para jugar el juego, `plot_learned_strategies` para visualizar las estrategias aprendidas, y `plot_agent_results` para visualizar los resultados de los agentes (ganas, perdidas y empates). 
Estas funciones utilizan bibliotecas como tqdm para la barra de progreso y matplotlib para la visualización.

## Observaciones y Resultados

### Aprendizaje contra Estrategias Sesgadas

Una de las primeras observaciones que se realizaron fue cómo los agentes de Fictitious Play (FP) y Regret Matching (RM) se adaptan a estrategias sesgadas. En particular, se quería ver si estos agentes podían "aprender a ganar" contra agentes como `RandomAgentBias` y `AlwaysRock`, que tienen una fuerte preferencia por ciertas acciones. Los resultados confirmaron que tanto FP como RM pudieron adaptarse efectivamente y explotar las debilidades de estos agentes sesgados en los tres juegos experimentados: RPS, MP y Blotto.

### Convergencia a Equilibrios de Nash

#### Piedra, Papel o Tijera (RPS)

En el juego de Piedra, Papel o Tijera, se observaron varios patrones interesantes:

1. **Regret Matching vs Regret Matching (RM vs RM)**: En este escenario, ambos agentes convergieron al equilibrio de Nash mixto de (1/3, 1/3, 1/3), lo cual es el resultado esperado y confirma la eficacia del algoritmo RM en encontrar equilibrios de Nash.

2. **Regret Matching vs Fictitious Play (RM vs FP)**: Similar al caso anterior, ambos agentes alcanzaron el equilibrio de Nash de (1/3, 1/3, 1/3), lo que demuestra la compatibilidad y robustez de estos dos algoritmos de aprendizaje.

3. **Regret Matching vs Random (RM vs Random)** y **Fictitious Play vs Random (FP vs Random)**: En ambos casos, se observó una tendencia hacia una acción particular en lugar de converger al equilibrio de Nash. Este comportamiento es intrigante y podría requerir una investigación más profunda.

4. **Fictitious Play vs Fictitious Play (FP vs FP)**: Aunque ambos agentes adoptaron una estrategia mixta de (1/3, 1/3, 1/3), curiosamente, nunca empataron entre ellos; siempre hubo un ganador y un perdedor. Este fenómeno es bastante inusual y no tiene una explicación clara en este momento.

#### Matching Pennies (MP)

En el juego de Matching Pennies, se observaron los siguientes patrones:

1. **RM vs RM** y **RM vs Random**: Ambos escenarios mostraron una convergencia hacia el equilibrio de Nash de (1/2, 1/2), lo cual es consistente con la teoría de juegos.

2. **Random vs Fictitious Play (Random vs FP)**: Aquí, el agente de Fictitious Play adoptó una estrategia pura de siempre elegir "head". Este comportamiento es notable y podría indicar una explotación de la aleatoriedad del agente Random.

3. **Fictitious Play vs Regret Matching (FP vs RM)** y **Fictitious Play vs Fictitious Play (FP vs FP)**: En ambos casos, los agentes convergieron al equilibrio de Nash de (1/2, 1/2).


## Interacción Usuario-Agente

Para aquellos interesados en experimentar cómo se siente jugar contra estos agentes de aprendizaje, se ha incluido una sección al final de la notebook `RPS.ipynb` que permite enfrentarse a cualquiera de los agentes implementados. El siguiente fragmento de código muestra cómo configurar un juego de Piedra, Papel o Tijera donde el usuario juega contra un agente de Regret Matching:

```python
# Si quieres jugar contra el agente
g.reset()
my_agents = {}
my_agents[g.agents[0]] = RegretMatching(g, g.agents[0])
my_agents[g.agents[1]] = HumanAgent(g, g.agents[1])
play_game(g, my_agents, 10)
```

Esta funcionalidad proporciona una forma interactiva de entender mejor cómo se comportan estos agentes y qué tan efectivos pueden ser en un entorno de juego real.



## Conclusión

Los experimentos confirmaron la eficacia de los algoritmos de Fictitious Play y Regret Matching en una variedad de escenarios. No solo pudieron adaptarse y explotar estrategias sesgadas, sino que también mostraron una notable capacidad para converger a equilibrios de Nash, tanto en estrategias mixtas como puras. Estos resultados subrayan el potencial de estos algoritmos para aplicaciones en entornos más complejos y dinámicos.

from pommerman.agents import BaseAgent, RandomAgent, SimpleAgent

from pommerlution.agents.dqn_agent import DQNAgent


class AgentFactory():
    def __init__(self) -> None:
        pass

    def get_agent(self, agent: str) -> BaseAgent:
        """
        Create an return an agent corresponding to `agent`.

        :param agent: String mapping to an agent. Can be one of
            "random_agent", "simple_agent" or "dqn_agent".
        """
        if agent == "random_agent":
            return RandomAgent()
        elif agent == "simple_agent":
            return SimpleAgent()
        elif agent == "dqn_agent":
            return DQNAgent()
        else:
            raise ValueError("Agent {agent} is unsupported")

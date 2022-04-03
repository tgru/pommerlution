from gym.spaces import Discrete
from pommerman.agents import BaseAgent

from pommerlution.models.dqn import DQN


class DQNAgent(BaseAgent):
    """
    A DQN based agent for the Pommerman environment.
    """

    def __init__(self, model: DQN = None):
        """
        Setup an agent based on a DQN model.

        :param model: A DQN model used for action selection
        """
        super(DQNAgent, self).__init__()

        self.model = model

    def act(self, obs: dict, action_space: Discrete):
        """
        Return the action to execute based on `obs`.

        :param obs: Environment observation to act on
        :param action_space: Number of actions expected by the
            environment. Currently unused.
        """
        return self.model.policy(obs)
